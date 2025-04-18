package usecase

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/xuri/excelize/v2"

	// "gorm.io/datatypes" // Removed - Not needed as SchemaDefinition is []FieldDefinition

	"golang-microservices-boilerplate/pkg/core/logger"
	"golang-microservices-boilerplate/pkg/core/types"
	waterPb "golang-microservices-boilerplate/proto/water-quality-service"
	"golang-microservices-boilerplate/services/water-quality-service/internal/entity"
	"golang-microservices-boilerplate/services/water-quality-service/internal/repository"
)

// ImportUseCase defines the interface for data import operations.
type ImportUseCase interface {
	ImportData(ctx context.Context, reader io.Reader, filename string, filetype string) (*waterPb.UploadDataResponse, error)
	GetDataSourceSchema(ctx context.Context, schemaID uuid.UUID) (*entity.DataSourceSchema, error)
	ListDataSourceSchemas(ctx context.Context, opts types.FilterOptions) (*types.PaginationResult[entity.DataSourceSchema], error)
}

type importService struct {
	stationRepo   repository.StationRepository
	dataPointRepo repository.DataPointRepository
	schemaRepo    repository.DataSourceSchemaRepository
	logger        logger.Logger
	// indicatorRepo repository.IndicatorRepository // Removed indicatorRepo
}

// NewImportService creates a new data import use case service.
func NewImportService(
	stationRepo repository.StationRepository,
	dataPointRepo repository.DataPointRepository,
	schemaRepo repository.DataSourceSchemaRepository,
	// indicatorRepo repository.IndicatorRepository, // Removed indicatorRepo
	logger logger.Logger,
) ImportUseCase {
	return &importService{
		stationRepo:   stationRepo,
		dataPointRepo: dataPointRepo,
		schemaRepo:    schemaRepo,
		// indicatorRepo: indicatorRepo, // Removed indicatorRepo
		logger: logger,
	}
}

// ImportData implements the core data import logic.
func (s *importService) ImportData(ctx context.Context, reader io.Reader, filename string, filetype string) (*waterPb.UploadDataResponse, error) {
	s.logger.Info("Starting data import", "filename", filename, "type", filetype)

	// 1. Determine/Load/Create DataSourceSchema
	// Need a TeeReader to allow reading headers for schema inference without consuming the reader
	var buf strings.Builder
	teeReader := io.TeeReader(reader, &buf)

	schema, err := s.getOrCreateSchema(ctx, filename, filetype, teeReader) // Pass teeReader here
	if err != nil {
		s.logger.Error("Failed to get or create schema", "error", err)
		return &waterPb.UploadDataResponse{
			Message: "Failed to process data schema",
			Errors:  []string{err.Error()},
		}, err
	}

	// IMPORTANT: Reset the original reader using the buffered content because teeReader consumed it
	// Now 'reader' points to the original content read into the buffer.
	reader = strings.NewReader(buf.String())

	// 2. Select Parser based on filetype
	var records [][]string
	var parseErr error

	switch entity.SourceType(filetype) {
	case entity.SourceTypeCSV:
		records, parseErr = s.parseCSV(reader)
	case entity.SourceTypeExcel:
		records, parseErr = s.parseExcel(reader)
	case entity.SourceTypeJSON:
		records, parseErr = s.parseJSON(reader)
	default:
		parseErr = fmt.Errorf("unsupported file type: %s", filetype)
	}

	if parseErr != nil {
		s.logger.Error("Failed to parse file", "error", parseErr, "filename", filename)
		return &waterPb.UploadDataResponse{
			Message: "Failed to parse file",
			Errors:  []string{parseErr.Error()},
		}, parseErr
	}

	// 3. Process Records using the Schema
	processedCount, failedCount, processingErrors := s.processRecordBatch(ctx, records, schema, filename)

	// 4. Construct Response
	response := &waterPb.UploadDataResponse{
		Message:            fmt.Sprintf("Import completed for %s.", filename),
		RecordsProcessed:   processedCount,
		RecordsFailed:      failedCount,
		DataSourceSchemaId: schema.ID.String(),
		Errors:             processingErrors,
	}

	if failedCount == 0 && len(processingErrors) == 0 {
		response.Message = fmt.Sprintf("Import successful for %s. Processed %d records.", filename, processedCount)
	} else {
		response.Message = fmt.Sprintf("Import completed with errors for %s. Processed: %d, Failed: %d.", filename, processedCount, failedCount)
	}

	s.logger.Info("Data import finished", "filename", filename, "processed", processedCount, "failed", failedCount)
	return response, nil
}

// GetDataSourceSchema retrieves a schema by ID
func (s *importService) GetDataSourceSchema(ctx context.Context, schemaID uuid.UUID) (*entity.DataSourceSchema, error) {
	return s.schemaRepo.FindByID(ctx, schemaID)
}

// ListDataSourceSchemas lists available schemas
func (s *importService) ListDataSourceSchemas(ctx context.Context, opts types.FilterOptions) (*types.PaginationResult[entity.DataSourceSchema], error) {
	return s.schemaRepo.FindAll(ctx, opts)
}

// Helper Functions

// parseCSV parses CSV data into a slice of string slices (rows of columns)
func (s *importService) parseCSV(r io.Reader) ([][]string, error) {
	csvReader := csv.NewReader(r)
	csvReader.LazyQuotes = true
	csvReader.TrimLeadingSpace = true
	records, err := csvReader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("error reading CSV: %w", err)
	}
	if len(records) < 2 {
		return nil, fmt.Errorf("CSV file must contain a header row and at least one data row")
	}
	return records, nil
}

// parseExcel parses Excel data into a slice of string slices (rows of columns)
func (s *importService) parseExcel(r io.Reader) ([][]string, error) {
	xlFile, err := excelize.OpenReader(r)
	if err != nil {
		return nil, fmt.Errorf("error opening Excel file: %w", err)
	}
	defer xlFile.Close()
	sheets := xlFile.GetSheetList()
	if len(sheets) == 0 {
		return nil, fmt.Errorf("excel file contains no sheets")
	}
	sheetName := sheets[0]
	rows, err := xlFile.GetRows(sheetName)
	if err != nil {
		return nil, fmt.Errorf("error reading Excel sheet '%s': %w", sheetName, err)
	}
	if len(rows) < 2 {
		return nil, fmt.Errorf("excel sheet must contain a header row and at least one data row")
	}
	// Clean coordinate formats if necessary
	for i, row := range rows {
		for j, cell := range row {
			if strings.Contains(cell, "\n") && (strings.Contains(cell, ",") || strings.Contains(cell, ".")) {
				rows[i][j] = strings.ReplaceAll(cell, "\n", " ")
			}
		}
	}
	return rows, nil
}

// parseJSON parses JSON data into a slice of string slices (to match CSV/Excel format)
func (s *importService) parseJSON(r io.Reader) ([][]string, error) {
	var jsonData []map[string]interface{}
	decoder := json.NewDecoder(r)
	if err := decoder.Decode(&jsonData); err != nil {
		return nil, fmt.Errorf("error decoding JSON: %w", err)
	}
	if len(jsonData) == 0 {
		return nil, fmt.Errorf("JSON file must contain at least one record")
	}
	headers := make(map[string]int)
	headerSlice := []string{}
	for _, record := range jsonData {
		for key := range record {
			if _, exists := headers[key]; !exists {
				headers[key] = len(headerSlice)
				headerSlice = append(headerSlice, key)
			}
		}
	}
	result := make([][]string, 0, len(jsonData)+1)
	result = append(result, headerSlice)
	for _, record := range jsonData {
		row := make([]string, len(headerSlice))
		for i, header := range headerSlice {
			if val, ok := record[header]; ok {
				row[i] = fmt.Sprintf("%v", val)
			} else {
				row[i] = ""
			}
		}
		result = append(result, row)
	}
	return result, nil
}

// getOrCreateSchema loads an existing schema or creates a new one based on the data
func (s *importService) getOrCreateSchema(ctx context.Context, filename string, filetype string, reader io.Reader) (*entity.DataSourceSchema, error) {
	schemaName := strings.TrimSuffix(filename, "."+filetype)
	sourceIdentifier := filename
	sourceType := entity.SourceType(filetype)

	schema, err := s.schemaRepo.FindByNameAndSource(ctx, schemaName, sourceIdentifier, sourceType)
	if err == nil && schema != nil {
		s.logger.Info("Found existing schema", "schema_id", schema.ID)
		if _, copyErr := io.Copy(io.Discard, reader); copyErr != nil {
			s.logger.Warn("Error discarding reader after finding schema", "error", copyErr)
		}
		return schema, nil
	} else if err != nil {
		s.logger.Warn("Error checking for existing schema, proceeding to infer", "error", err)
	}

	s.logger.Info("No existing schema found, attempting to infer from headers", "filename", filename)
	var headers []string
	var headerParseErr error

	switch sourceType {
	case entity.SourceTypeCSV:
		records, err := s.parseCSV(reader)
		if err != nil {
			headerParseErr = fmt.Errorf("error reading CSV headers: %w", err)
		} else if len(records) > 0 {
			headers = records[0]
		}
	case entity.SourceTypeExcel:
		records, err := s.parseExcel(reader)
		if err != nil {
			headerParseErr = fmt.Errorf("error reading Excel headers: %w", err)
		} else if len(records) > 0 {
			headers = records[0]
		}
	case entity.SourceTypeJSON:
		records, err := s.parseJSON(reader)
		if err != nil {
			headerParseErr = fmt.Errorf("error reading JSON headers: %w", err)
		} else if len(records) > 0 {
			headers = records[0]
		}
	default:
		headerParseErr = fmt.Errorf("unsupported source type for schema inference: %s", sourceType)
	}

	if headerParseErr != nil {
		return nil, headerParseErr
	}
	if len(headers) == 0 {
		return nil, fmt.Errorf("could not extract headers from file to infer schema")
	}

	fieldDefs := s.inferSchemaFromHeaders(headers)

	newSchema := &entity.DataSourceSchema{
		Name:             schemaName,
		SourceIdentifier: sourceIdentifier,
		SourceType:       sourceType,
		Description:      "Auto-generated schema from " + filename,
		SchemaDefinition: fieldDefs, // Assign the slice directly
	}

	if err := s.schemaRepo.Create(ctx, newSchema); err != nil {
		return nil, fmt.Errorf("error creating schema: %w", err)
	}
	s.logger.Info("Successfully created new schema", "schema_id", newSchema.ID)
	return newSchema, nil
}

// inferSchemaFromHeaders creates FieldDefinition objects from headers
func (s *importService) inferSchemaFromHeaders(headers []string) []entity.FieldDefinition {
	fieldDefs := make([]entity.FieldDefinition, 0, len(headers))
	for _, header := range headers {
		fieldDef := entity.FieldDefinition{
			SourceName:   header,
			DataType:     entity.DataTypeString,
			TargetEntity: entity.TargetEntityIgnore,
			IsRequired:   false,
		}
		lowerHeader := strings.ToLower(header)

		// Station fields
		if contains(lowerHeader, []string{"station", "location", "site", "điểm quan trắc", "monitoringlocationidentifier"}) {
			fieldDef.TargetEntity = entity.TargetEntityStation
			fieldDef.TargetField = "Name"
		} else if contains(lowerHeader, []string{"lat", "latitude", "vĩ độ", "latitudemeasure_wgs84"}) {
			fieldDef.TargetEntity = entity.TargetEntityStation
			fieldDef.TargetField = "Latitude"
			fieldDef.DataType = entity.DataTypeFloat
		} else if contains(lowerHeader, []string{"lon", "long", "longitude", "kinh độ", "longitudemeasure_wgs84"}) {
			fieldDef.TargetEntity = entity.TargetEntityStation
			fieldDef.TargetField = "Longitude"
			fieldDef.DataType = entity.DataTypeFloat
		} else if contains(lowerHeader, []string{"tọa độ", "coordinate", "coordinates"}) {
			fieldDef.TargetEntity = entity.TargetEntityStation
			fieldDef.TargetField = "Coordinates"
			fieldDef.DataType = entity.DataTypeCoordinate
		} else if contains(lowerHeader, []string{"country", "quốc gia"}) {
			fieldDef.TargetEntity = entity.TargetEntityStation
			fieldDef.TargetField = "Country"
		} else if contains(lowerHeader, []string{"province", "tỉnh"}) {
			fieldDef.TargetEntity = entity.TargetEntityStation
			fieldDef.TargetField = "Province"
		} else if contains(lowerHeader, []string{"district", "huyện"}) {
			fieldDef.TargetEntity = entity.TargetEntityStation
			fieldDef.TargetField = "District"
		} else if contains(lowerHeader, []string{"time", "date", "monitoring_time", "timestamp", "ngày quan trắc", "monitoringdate"}) {
			fieldDef.TargetEntity = entity.TargetEntityDataPoint
			fieldDef.TargetField = "MonitoringTime"
			fieldDef.DataType = entity.DataTypeDateTime
		} else if contains(lowerHeader, []string{"wqi", "quality index"}) {
			fieldDef.TargetEntity = entity.TargetEntityDataPoint
			fieldDef.TargetField = "WQI"
			fieldDef.DataType = entity.DataTypeFloat
		} else if contains(lowerHeader, []string{"observation_type"}) {
			fieldDef.TargetEntity = entity.TargetEntityDataPoint
			fieldDef.TargetField = "ObservationType"
			fieldDef.DataType = entity.DataTypeString
		} else if contains(lowerHeader, []string{"sourceprovider", "source"}) {
			fieldDef.TargetEntity = entity.TargetEntityDataPoint
			fieldDef.TargetField = "Source"
			fieldDef.DataType = entity.DataTypeString
		} else {
			// Default to Indicator/Feature if not matched above
			fieldDef.TargetEntity = entity.TargetEntityIndicator
			fieldDef.TargetField = ""

			// Extract unit from header if present (e.g., "TSS (mg/L)")
			if strings.Contains(header, " (") && strings.HasSuffix(header, ")") {
				parts := strings.SplitN(header, " (", 2)
				if len(parts) == 2 {
					fieldDef.SourceName = strings.TrimSpace(parts[0]) // Use name before unit
					fieldDef.Unit = strings.TrimSuffix(parts[1], ")")
				}
			} else if lowerHeader == "value" {
				fieldDef.SourceName = "Value"
				fieldDef.DataType = entity.DataTypeFloat
				fieldDef.Purpose = entity.PurposeAnalysis
				fieldDef.TargetField = "_GenericValue_"
			} else if lowerHeader == "indicatorsname" {
				fieldDef.TargetEntity = entity.TargetEntityIgnore
				fieldDef.TargetField = "_IndicatorName_"
			} else {
				if fieldDef.SourceName == header {
					// Already set by default
				}
			}

			// Guess data type and purpose based on typical indicators
			featureNameLower := strings.ToLower(fieldDef.SourceName)
			if fieldDef.TargetField != "_GenericValue_" && fieldDef.TargetField != "_IndicatorName_" {
				// Check for bacterial indicators first
				if contains(featureNameLower, []string{"coliform", "aeromonas", "edwardsiella"}) {
					// Could be numeric count or boolean/text
					if contains(featureNameLower, []string{"tổng số", "count"}) {
						fieldDef.DataType = entity.DataTypeFloat // Assume float for counts
					} else {
						fieldDef.DataType = entity.DataTypeBoolean // Assume boolean otherwise ("Âm tính"/"Dương tính")
					}
					fieldDef.Purpose = entity.PurposeAnalysis
				} else if contains(featureNameLower, []string{"ph", "nhiệt độ", "temperature", "do", "conductivity", "oxygen", "cod", "bod", "tss", "no2", "nh4", "po4", "h2s", "độ dẫn", "độ kiềm", "n-no2", "n-nh4", "p-po4", "ec", "ah"}) { // Keep 'ah' here for now, but it's checked after aeromonas
					fieldDef.DataType = entity.DataTypeFloat
					fieldDef.Purpose = entity.PurposeAnalysis // Default purpose, refine later
				} else if contains(featureNameLower, []string{"chất lượng nước", "quality", "khuyến cáo", "recommendations", "chỉ tiêu vượt ngưỡng"}) {
					fieldDef.DataType = entity.DataTypeText
					fieldDef.Purpose = entity.PurposeDisplay
				}
			}
		}
		fieldDefs = append(fieldDefs, fieldDef)
	}
	return fieldDefs
}

// contains checks if any of the terms are in the target string
func contains(target string, terms []string) bool {
	for _, term := range terms {
		if strings.Contains(target, term) {
			return true
		}
	}
	return false
}

// processRecordBatch processes a batch of records using the schema
func (s *importService) processRecordBatch(ctx context.Context, records [][]string, schema *entity.DataSourceSchema, source string) (int64, int64, []string) {
	if len(records) < 2 {
		return 0, 0, []string{"Not enough records to process (need header + data)"}
	}

	// Use field definitions directly from the schema struct
	fieldDefs := schema.SchemaDefinition
	if len(fieldDefs) == 0 {
		return 0, int64(len(records) - 1), []string{"Schema definition is empty"}
	}

	// --- Refine purposes based on specific needs (Example) ---
	// This section might need adjustment based on how you definitively identify
	// prediction vs analysis fields in your actual schema or input data.
	predictionFields := map[string]bool{
		"ph":                true,
		"do":                true,
		"conductivity":      true,
		"độ dẫn":            true,
		"n-no2":             true,
		"n-nh4":             true,
		"p-po4":             true,
		"tss":               true,
		"cod":               true,
		"aeromonas tổng số": true,
	}
	displayFields := map[string]bool{
		"khuyến cáo":           true,
		"recommendations":      true,
		"chất lượng nước":      true,
		"chỉ tiêu vượt ngưỡng": true,
	}

	for i, def := range fieldDefs {
		if def.TargetEntity == entity.TargetEntityIndicator { // Features marked as Indicator target
			lowerName := strings.ToLower(def.SourceName)
			if predictionFields[lowerName] {
				fieldDefs[i].Purpose = entity.PurposePrediction
			} else if displayFields[lowerName] {
				fieldDefs[i].Purpose = entity.PurposeDisplay
			} else {
				// Default remaining features to Analysis if not specified
				if fieldDefs[i].Purpose == "" {
					fieldDefs[i].Purpose = entity.PurposeAnalysis
				}
			}
		}
	}
	// --- End Refinement ---

	headers := records[0]
	fieldIndex := make(map[string]int)
	indicatorNameColIdx := -1
	genericValueColIdx := -1
	for i, header := range headers {
		fieldIndex[header] = i
		// Find special columns for flat indicator format
		for _, def := range fieldDefs {
			if def.SourceName == header {
				if def.TargetField == "_IndicatorName_" {
					indicatorNameColIdx = i
				} else if def.TargetField == "_GenericValue_" {
					genericValueColIdx = i
				}
				break
			}
		}
	}

	stationsMap := make(map[string]*entity.Station)
	var dataPointsToCreate []*entity.DataPoint
	dpToStationKey := make(map[*entity.DataPoint]string)

	var processingErrors []string
	processedCount := int64(0)
	failedCount := int64(0)

	for rowIdx, row := range records[1:] {
		if len(row) == 0 || isEmptyRow(row) {
			continue
		}
		if len(row) < len(headers) {
			paddedRow := make([]string, len(headers))
			copy(paddedRow, row)
			for i := len(row); i < len(headers); i++ {
				paddedRow[i] = ""
			}
			row = paddedRow
			// processingErrors = append(processingErrors, fmt.Sprintf("Row %d: Warning - padded short row", rowIdx+2))
		}

		station, err := s.extractStation(fieldDefs, row, headers, source)
		if err != nil {
			processingErrors = append(processingErrors, fmt.Sprintf("Row %d: Error extracting station: %v", rowIdx+2, err))
			failedCount++
			continue
		}
		stationKey := fmt.Sprintf("%.6f|%.6f", station.Latitude, station.Longitude)
		if _, found := stationsMap[stationKey]; !found {
			stationsMap[stationKey] = station
		}

		dataPoint, err := s.extractDataPoint(fieldDefs, row, headers, uuid.Nil, source)
		if err != nil {
			processingErrors = append(processingErrors, fmt.Sprintf("Row %d: Error extracting data point: %v", rowIdx+2, err))
			failedCount++
			continue
		}

		// Extract features
		features, err := s.extractFeatures(fieldDefs, row, headers, dataPoint, source, indicatorNameColIdx, genericValueColIdx)
		if err != nil {
			processingErrors = append(processingErrors, fmt.Sprintf("Row %d: Error extracting features: %v", rowIdx+2, err))
			// Decide if row fails based on feature errors
		}
		dataPoint.Features = features
		dataPoint.DataSourceSchemaID = schema.ID // Assign the schema ID used for this import

		dpToStationKey[dataPoint] = stationKey
		dataPointsToCreate = append(dataPointsToCreate, dataPoint)
		processedCount++
	}

	stationList := make([]*entity.Station, 0, len(stationsMap))
	stationKeyToID := make(map[string]uuid.UUID)
	for _, station := range stationsMap {
		stationList = append(stationList, station)
	}

	if len(stationList) > 0 {
		createdOrFoundStations, err := s.stationRepo.CreateMany(ctx, stationList)
		if err != nil {
			processingErrors = append(processingErrors, fmt.Sprintf("Error batch saving/finding stations: %v", err))
			return 0, int64(len(records) - 1), processingErrors
		}
		for _, station := range createdOrFoundStations {
			key := fmt.Sprintf("%.6f|%.6f", station.Latitude, station.Longitude)
			stationKeyToID[key] = station.ID
		}
		s.logger.Info("Batch processed stations (created or found)", "count", len(createdOrFoundStations))
	}

	var validDataPointsToCreate []*entity.DataPoint
	finalFailedCount := failedCount // Start with failures from extraction phase
	if len(dataPointsToCreate) > 0 {
		for _, dp := range dataPointsToCreate {
			if stationKey, dpOk := dpToStationKey[dp]; dpOk {
				if stationID, keyOk := stationKeyToID[stationKey]; keyOk {
					dp.StationID = stationID
					validDataPointsToCreate = append(validDataPointsToCreate, dp)
				} else {
					processingErrors = append(processingErrors, fmt.Sprintf("Row %d: Could not find ID for station key '%s' required by data point at %s", findRowIndex(records, dp.MonitoringTime, stationKey)+2, stationKey, dp.MonitoringTime))
					finalFailedCount++ // Increment failed count
				}
			} else {
				processingErrors = append(processingErrors, fmt.Sprintf("Row %d: Internal error - Data point missing station key association at %s", findRowIndex(records, dp.MonitoringTime, "")+2, dp.MonitoringTime))
				finalFailedCount++ // Increment failed count
			}
		}

		if len(validDataPointsToCreate) > 0 {
			createdOrFoundDataPoints, dpCreateErr := s.dataPointRepo.CreateMany(ctx, validDataPointsToCreate)
			if dpCreateErr != nil {
				processingErrors = append(processingErrors, fmt.Sprintf("Error batch saving/finding data points: %v", dpCreateErr))
				// Adjust counts - Assume all attempted in this batch failed
				finalFailedCount += int64(len(validDataPointsToCreate))
				// processedCount is harder to adjust accurately here, leave as is
			} else {
				s.logger.Info("Batch processed data points (created or found)", "count", len(createdOrFoundDataPoints))
				// Potentially adjust failed count if CreateMany handled some conflicts gracefully
			}
		}
	}

	// Ensure counts are consistent
	totalRowsProcessed := int64(len(records) - 1)
	finalProcessedCount := totalRowsProcessed - finalFailedCount
	if finalProcessedCount < 0 {
		finalProcessedCount = 0
	}

	return finalProcessedCount, finalFailedCount, processingErrors
}

// Helper to check if a row is effectively empty
func isEmptyRow(row []string) bool {
	for _, cell := range row {
		if strings.TrimSpace(cell) != "" {
			return false
		}
	}
	return true
}

// Helper to find row index (for error reporting, potentially slow)
func findRowIndex(records [][]string, monitoringTime time.Time, stationKey string) int {
	// Simplified search - this won't be perfectly accurate without more context
	timeStr := monitoringTime.Format("2006-01-02") // Example format match
	for i, row := range records[1:] {
		// Rudimentary check, improve if needed
		rowTimeStr := ""
		// rowStationKey := "" // Removed unused variable
		// Try to find time and station info in row based on common patterns
		// This needs to be more robust based on schema
		if len(row) > 5 {
			rowTimeStr = row[5]
		} // Guessing time column index

		if strings.Contains(rowTimeStr, timeStr) { // && strings.Contains(rowStationKey, stationKey) { // stationKey match is harder
			return i + 1 // Return 1-based index relative to data rows
		}
	}
	return -1 // Indicate not found
}

// extractStation creates a Station entity from a row based on the schema
func (s *importService) extractStation(fieldDefs []entity.FieldDefinition, row, headers []string, _ string) (*entity.Station, error) {
	station := &entity.Station{}
	foundName, foundLat, foundLon := false, false, false
	var province, district string

	for _, fieldDef := range fieldDefs {
		if fieldDef.TargetEntity != entity.TargetEntityStation {
			continue
		}
		colIdx := -1
		for j, header := range headers {
			if header == fieldDef.SourceName {
				colIdx = j
				break
			}
		}
		if colIdx == -1 || colIdx >= len(row) {
			if fieldDef.IsRequired {
				return nil, fmt.Errorf("required station field '%s' missing in headers or row", fieldDef.SourceName)
			}
			continue
		}
		value := strings.TrimSpace(row[colIdx])
		if value == "" {
			if fieldDef.IsRequired {
				return nil, fmt.Errorf("required station field '%s' has empty value", fieldDef.SourceName)
			}
			continue
		}

		switch fieldDef.TargetField {
		case "Name":
			station.Name = value
			foundName = true
		case "Latitude":
			lat, err := strconv.ParseFloat(strings.ReplaceAll(value, ",", "."), 64)
			if err != nil {
				return nil, fmt.Errorf("invalid latitude value '%s' for field '%s': %w", value, fieldDef.SourceName, err)
			}
			station.Latitude = lat
			foundLat = true
		case "Longitude":
			lon, err := strconv.ParseFloat(strings.ReplaceAll(value, ",", "."), 64)
			if err != nil {
				return nil, fmt.Errorf("invalid longitude value '%s' for field '%s': %w", value, fieldDef.SourceName, err)
			}
			station.Longitude = lon
			foundLon = true
		case "Coordinates":
			// Handle formats like "lat,decimal\nlon,decimal" or "lat.decimal\nlon.decimal"
			// Replace newline with a space, then comma with period for parsing
			coordStr := strings.ReplaceAll(value, "\n", " ")
			coordStr = strings.ReplaceAll(coordStr, ",", ".")
			parts := strings.Fields(coordStr) // Split by whitespace
			if len(parts) >= 2 {
				// Assume first part is latitude, second is longitude
				latStr := parts[0]
				lonStr := parts[1]
				parsedLat, errLat := strconv.ParseFloat(latStr, 64)
				parsedLon, errLon := strconv.ParseFloat(lonStr, 64)
				if errLat == nil && errLon == nil {
					station.Latitude = parsedLat
					station.Longitude = parsedLon
					foundLat = true
					foundLon = true
				} else {
					// Construct a more informative error message
					errors := []string{}
					if errLat != nil {
						errors = append(errors, fmt.Sprintf("latitude parse error: %v", errLat))
					}
					if errLon != nil {
						errors = append(errors, fmt.Sprintf("longitude parse error: %v", errLon))
					}
					return nil, fmt.Errorf("invalid coordinate value '%s' (processed as '%s') for field '%s': %s", value, coordStr, fieldDef.SourceName, strings.Join(errors, "; "))
				}
			} else {
				return nil, fmt.Errorf("invalid coordinate format '%s' (processed as '%s') for field '%s': expected two numeric parts after processing", value, coordStr, fieldDef.SourceName)
			}
		case "Country":
			station.Country = value
		case "Province":
			province = value
		case "District":
			district = value
		case "Location":
			station.Location = value
		}
	}

	if station.Location == "" && (province != "" || district != "") {
		locationParts := []string{}
		if district != "" {
			locationParts = append(locationParts, district)
		}
		if province != "" {
			locationParts = append(locationParts, province)
		}
		station.Location = strings.Join(locationParts, ", ")
	}

	if !foundName {
		return nil, fmt.Errorf("station name could not be extracted (check schema mapping for 'Name')")
	}
	if !foundLat || !foundLon {
		return nil, fmt.Errorf("station latitude/longitude could not be extracted (check schema mapping for 'Latitude'/'Longitude' or 'Coordinates')")
	}
	return station, nil
}

// extractDataPoint creates a DataPoint entity from a row based on the schema (without features)
func (s *importService) extractDataPoint(fieldDefs []entity.FieldDefinition, row, headers []string, stationID uuid.UUID, source string) (*entity.DataPoint, error) {
	dataPoint := &entity.DataPoint{
		StationID: stationID, // Will be updated later
		Source:    source,
	}
	foundTime := false

	for _, fieldDef := range fieldDefs {
		if fieldDef.TargetEntity != entity.TargetEntityDataPoint {
			continue
		}
		colIdx := -1
		for j, header := range headers {
			if header == fieldDef.SourceName {
				colIdx = j
				break
			}
		}
		if colIdx == -1 || colIdx >= len(row) {
			if fieldDef.IsRequired {
				return nil, fmt.Errorf("required data point field '%s' missing", fieldDef.SourceName)
			}
			continue
		}
		value := strings.TrimSpace(row[colIdx])
		if value == "" {
			if fieldDef.IsRequired {
				return nil, fmt.Errorf("required data point field '%s' empty", fieldDef.SourceName)
			}
			continue
		}

		switch fieldDef.TargetField {
		case "MonitoringTime":
			t, err := parseTime(value)
			if err != nil {
				return nil, fmt.Errorf("invalid monitoring time value '%s' for field '%s': %w", value, fieldDef.SourceName, err)
			}
			dataPoint.MonitoringTime = t
			foundTime = true
		case "WQI":
			valueNormalized := strings.ReplaceAll(value, ",", ".")
			wqi, err := strconv.ParseFloat(valueNormalized, 64)
			if err != nil {
				// Allow WQI parsing failure, just won't set the value
				s.logger.Warn("Invalid WQI value, skipping", "value", value, "field", fieldDef.SourceName, "error", err)
			} else {
				dataPoint.WQI = &wqi // Assign address
			}
		case "ObservationType":
			ot := entity.ObservationType(value)
			if ot != entity.Actual && ot != entity.Interpolation && ot != entity.Predicted && ot != entity.RealtimeMonitoring {
				s.logger.Warn("Unknown observation type, using default", "value", value, "field", fieldDef.SourceName)
				ot = entity.Actual
			}
			dataPoint.ObservationType = ot
		case "Source": // Allow overriding the default source (filename)
			dataPoint.Source = value
			// case "DataSourceSchemaID": // Should be set globally
		}
	}

	// Ensure a default ObservationType if none was provided or extracted
	if dataPoint.ObservationType == "" {
		s.logger.Warn("Observation type was empty or missing, defaulting to 'actual'", "stationID", dataPoint.StationID, "time", dataPoint.MonitoringTime, "source", source)
		dataPoint.ObservationType = entity.Actual
	}

	if !foundTime {
		return nil, fmt.Errorf("monitoring time is required but not found (check schema mapping for 'MonitoringTime')")
	}
	return dataPoint, nil
}

// parseTime attempts to parse a time string using various formats
func parseTime(value string) (time.Time, error) {
	formats := []string{
		"2006-01-02 15:04:05",  // Standard datetime
		"2006-01-02T15:04:05Z", // ISO 8601 UTC
		"2006-01-02",           // Date only
		"1/2/2006",             // Common US M/D/YYYY
		"1/2/06",               // Common US M/D/YY
		"01/02/2006",           // MM/DD/YYYY
		"01/02/06",             // MM/DD/YY
		"2/1/2006",             // Common EU D/M/YYYY
		"2/1/06",               // Common EU D/M/YY
		"02/01/2006",           // DD/MM/YYYY
		"02/01/06",             // DD/MM/YY
		"02/01/2006 15:04:05",  // DD/MM/YYYY HH:MM:SS (Vietnamese format with time)
		"Jan 2, 2006",          // Month Day, Year
		"2 Jan 2006",           // Day Month Year
		time.RFC3339,           // Includes timezone offset
		time.RFC3339Nano,
		time.RFC1123,
		time.RFC1123Z,
		time.RFC822,
		time.RFC822Z,
		time.ANSIC,
		time.UnixDate,
	}
	value = strings.TrimSpace(value)
	for _, format := range formats {
		if t, err := time.Parse(format, value); err == nil {
			return t, nil
		}
	}
	// Handle specific Vietnamese format dd/mm/yyyy if not caught by standard ones
	if strings.Count(value, "/") == 2 {
		parts := strings.Split(value, "/")
		if len(parts) == 3 {
			if len(parts[0]) <= 2 && len(parts[1]) <= 2 && len(parts[2]) == 4 {
				// Assume DD/MM/YYYY
				reformatted := fmt.Sprintf("%s-%s-%s", parts[2], parts[1], parts[0])
				if t, err := time.Parse("2006-01-02", reformatted); err == nil {
					return t, nil
				}
			}
		}
	}
	return time.Time{}, fmt.Errorf("could not parse time '%s' using known formats", value)
}

// extractFeatures creates DataPointFeature structs from a row based on the schema
// Updated signature to handle flat indicator format (indicatorNameColIdx, genericValueColIdx)
func (s *importService) extractFeatures(fieldDefs []entity.FieldDefinition, row, headers []string, parentDataPoint *entity.DataPoint, source string, indicatorNameColIdx, genericValueColIdx int) ([]entity.DataPointFeature, error) {
	var features []entity.DataPointFeature
	var multiErr error

	// Handle flat format (IndicatorsName, Value columns)
	if indicatorNameColIdx != -1 && genericValueColIdx != -1 && genericValueColIdx < len(row) && indicatorNameColIdx < len(row) {
		featureName := strings.TrimSpace(row[indicatorNameColIdx])
		valueStr := strings.TrimSpace(row[genericValueColIdx])

		if featureName != "" && valueStr != "" {
			// Find the corresponding FieldDefinition for this feature name (if any)
			var fieldDef *entity.FieldDefinition
			for i := range fieldDefs {
				if strings.EqualFold(fieldDefs[i].SourceName, featureName) {
					fieldDef = &fieldDefs[i]
					break
				}
			}

			feature := entity.DataPointFeature{
				Name:   featureName,
				Source: source,
			}

			// Determine DataType and Purpose from schema if found, otherwise default
			dt := entity.DataTypeFloat // Default to float for generic value
			purpose := entity.PurposeAnalysis
			if fieldDef != nil {
				dt = fieldDef.DataType
				purpose = fieldDef.Purpose
			}
			feature.Purpose = purpose

			// Parse value based on determined DataType
			err := parseAndSetFeatureValue(&feature, valueStr, dt, featureName)
			if err != nil {
				s.logger.Warn("Error parsing feature value (flat format)", "feature", featureName, "value", valueStr, "error", err)
				// multiErr = multierror.Append(multiErr, err) // Optional: collect errors
			} else {
				features = append(features, feature)
			}
		}
	} else {
		// Handle wide format (each feature is a column)
		for _, fieldDef := range fieldDefs {
			if fieldDef.TargetEntity != entity.TargetEntityIndicator {
				continue
			}

			colIdx := -1
			for j, header := range headers {
				if header == fieldDef.SourceName {
					colIdx = j
					break
				}
			}
			if colIdx == -1 || colIdx >= len(row) {
				continue
			}
			value := strings.TrimSpace(row[colIdx])
			if value == "" {
				continue
			}

			feature := entity.DataPointFeature{
				Name:    fieldDef.SourceName,
				Purpose: fieldDef.Purpose,
				Source:  source,
			}

			err := parseAndSetFeatureValue(&feature, value, fieldDef.DataType, fieldDef.SourceName)
			if err != nil {
				s.logger.Warn("Error parsing feature value (wide format)", "feature", fieldDef.SourceName, "value", value, "error", err)
				// multiErr = multierror.Append(multiErr, err) // Optional: collect errors
			} else {
				features = append(features, feature)
			}
		}
	}

	return features, multiErr
}

// parseAndSetFeatureValue is a helper to parse and set Value/TextualValue based on DataType
func parseAndSetFeatureValue(feature *entity.DataPointFeature, valueStr string, dataType entity.FieldDataType, featureName string) error {
	switch dataType {
	case entity.DataTypeFloat, entity.DataTypeInteger:
		valueNormalized := strings.ReplaceAll(valueStr, ",", ".")
		floatVal, err := strconv.ParseFloat(valueNormalized, 64)
		if err != nil {
			// Fallback to textual value if parsing fails
			textVal := valueStr
			feature.TextualValue = &textVal
			return fmt.Errorf("invalid numeric value '%s' for feature '%s': %w", valueStr, featureName, err)
		} else {
			feature.Value = &floatVal
		}
	case entity.DataTypeBoolean:
		lowerValue := strings.ToLower(valueStr)
		isTrue := lowerValue == "true" || valueStr == "1" || lowerValue == "yes" || lowerValue == "dương tính" || lowerValue == "có" || lowerValue == "positive"
		isFalse := lowerValue == "false" || valueStr == "0" || lowerValue == "no" || lowerValue == "âm tính" || lowerValue == "không" || lowerValue == "negative"

		// Always store original text
		textVal := valueStr
		feature.TextualValue = &textVal

		// Only set numeric Value if it clearly matches true/false
		if isTrue {
			floatVal := 1.0
			feature.Value = &floatVal
		} else if isFalse {
			floatVal := 0.0
			feature.Value = &floatVal
		}
		// If neither true nor false, we have already stored the text, so return nil (no error)

	case entity.DataTypeString, entity.DataTypeText, entity.DataTypeUnknown, entity.DataTypeCoordinate, entity.DataTypeDate, entity.DataTypeDateTime:
		textVal := valueStr
		feature.TextualValue = &textVal
	default:
		// Default to textual value for any other unhandled type
		textVal := valueStr
		feature.TextualValue = &textVal
		return fmt.Errorf("unhandled data type '%s' for feature '%s', stored as text", dataType, featureName)
	}
	return nil
}
