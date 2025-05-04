package controller

import (
	"context"
	"fmt"
	"io" // Required for streaming upload
	"net/http"
	"strings" // Added for checking empty strings

	// Added for http status code constants if needed by MapErrorToHttpStatus
	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"

	coreController "golang-microservices-boilerplate/pkg/core/controller" // Added for FilterOptions
	"golang-microservices-boilerplate/pkg/middleware"                     // Added for GetClaimsFromContext
	pb "golang-microservices-boilerplate/proto/user-service"
	"golang-microservices-boilerplate/services/user-service/internal/usecase"
)

// Ensure fileServer implements pb.FileServiceServer.
var _ pb.FileServiceServer = (*fileServer)(nil)

type fileServer struct {
	pb.UnimplementedFileServiceServer
	uc     usecase.FileUsecase
	mapper Mapper // Use the general Mapper
}

// NewFileServer creates a new gRPC file server instance.
func NewFileServer(uc usecase.FileUsecase, mapper Mapper) pb.FileServiceServer {
	return &fileServer{
		uc:     uc,
		mapper: mapper,
	}
}

// RegisterFileServiceServer registers the file service implementation.
func RegisterFileServiceServer(s *grpc.Server, uc usecase.FileUsecase, mapper Mapper) {
	server := NewFileServer(uc, mapper)
	pb.RegisterFileServiceServer(s, server)
}

// --- Implement FileServiceServer Methods ---

// Upload handles client-streaming file upload using an async pattern with pipes.
func (s *fileServer) Upload(stream pb.FileService_UploadServer) error {
	ctx := stream.Context() // Get context from stream
	// log := s.uc.GetLogger().With("method", "Upload") // Assuming logger is accessible

	// 1. Receive the first message (expecting Name)
	reqName, err := stream.Recv()
	if err != nil {
		// log.Error("Failed to receive first message (expected name)", "error", err)
		if err == io.EOF {
			return status.Error(codes.InvalidArgument, "Stream closed before receiving filename")
		}
		return status.Error(codes.Unavailable, "Failed to receive filename")
	}

	var fileName string
	switch v := reqName.RequestData.(type) {
	case *pb.UploadFileRequest_Name:
		fileName = v.Name
		if strings.TrimSpace(fileName) == "" {
			// log.Warn("Received empty filename")
			return status.Error(codes.InvalidArgument, "Filename cannot be empty")
		}
	default:
		// log.Warn("Received unexpected data instead of filename in first message", "type", fmt.Sprintf("%T", v))
		return status.Error(codes.InvalidArgument, "First message must contain filename")
	}
	// log = log.With("fileName", fileName) // Add filename to logger context

	// 2. Receive the second message (expecting Type)
	reqType, err := stream.Recv()
	if err != nil {
		// log.Error("Failed to receive second message (expected type)", "error", err)
		if err == io.EOF {
			return status.Error(codes.InvalidArgument, "Stream closed before receiving file type")
		}
		return status.Error(codes.Unavailable, "Failed to receive file type")
	}

	var mimeType string
	switch v := reqType.RequestData.(type) {
	case *pb.UploadFileRequest_Type:
		mimeType = v.Type
		if strings.TrimSpace(mimeType) == "" {
			// log.Warn("Received empty mime type")
			return status.Error(codes.InvalidArgument, "MIME type cannot be empty")
		}
	default:
		// log.Warn("Received unexpected data instead of mime type in second message", "type", fmt.Sprintf("%T", v))
		return status.Error(codes.InvalidArgument, "Second message must contain file type")
	}
	// log = log.With("mimeType", mimeType) // Add mime type to logger context
	// log.Info("Received metadata")

	// 3. Get Uploader ID from context claims
	uploaderID, err := middleware.GetUserIdFromGRPCContext(ctx)
	if err != nil {
		// log.Error("Failed to get user ID from context", "error", err)
		return status.Errorf(http.StatusUnauthorized, "%v", err)
	}
	// log = log.With("uploaderID", uploaderID.String()) // Add uploaderID to logger context

	// 4. Setup Pipe and channels for asynchronous processing
	pr, pw := io.Pipe()
	done := make(chan struct{})
	var processingErr error

	// 5. Start the processing goroutine (calls use case and handles response)
	go s.runUploadAndHandleResponse(ctx, pr, fileName, mimeType, uploaderID, done, &processingErr, stream)

	// 6. Start receiving file chunks (third and subsequent messages) and write them to the pipe
	var totalSize int64 = 0
	for {
		chunkReq, err := stream.Recv()
		if err == io.EOF {
			// Expected end of stream from client
			// log.Info("EOF received from client stream", "totalSize", totalSize)
			if err := pw.Close(); err != nil {
				// log.Error("Failed to close pipe writer cleanly", "error", err)
			}
			break // Exit the receive loop
		}
		if err != nil {
			// Error receiving from client stream
			// log.Error("Error receiving chunk from client stream", "error", err)
			closeErr := fmt.Errorf("client stream error: %w", err)
			pw.CloseWithError(closeErr)
			<-done
			if processingErr != nil {
				return processingErr
			}
			return status.Error(codes.Unavailable, closeErr.Error())
		}

		// Process the received message (expecting ChunkData)
		var chunkData []byte
		switch v := chunkReq.RequestData.(type) {
		case *pb.UploadFileRequest_ChunkData:
			chunkData = v.ChunkData
			if len(chunkData) == 0 {
				// log.Debug("Received empty chunk data, skipping write")
				continue // Skip writing empty chunks
			}
		default:
			// Received unexpected data type after metadata
			err = fmt.Errorf("received unexpected data type (%T) after metadata, expected ChunkData", v)
			// log.Error("Protocol error", "error", err)
			pw.CloseWithError(err)
			<-done
			if processingErr != nil {
				return processingErr
			}
			return status.Error(codes.InvalidArgument, err.Error())
		}

		// Write the chunk to the pipe
		n, writeErr := pw.Write(chunkData)
		if writeErr != nil {
			// Error writing to pipe (likely means processing goroutine failed and closed the reader)
			// log.Error("Error writing chunk to pipe", "error", writeErr)
			<-done               // Wait for processing goroutine to finish
			return processingErr // Return the error from the processing goroutine
		}
		totalSize += int64(n)
		// log.Debug("Wrote chunk to pipe", "bytes", n, "totalSize", totalSize)
	}

	// 7. Wait for the processing goroutine to complete
	<-done
	// log.Info("Processing finished", "finalError", processingErr)

	// 8. Return the final error status from the processing goroutine
	return processingErr
}

// runUploadAndHandleResponse handles calling the use case and sending the final gRPC response.
func (s *fileServer) runUploadAndHandleResponse(
	ctx context.Context,
	pr *io.PipeReader, // Read file data from here
	filename, mimeType string,
	uploaderID uuid.UUID,
	done chan<- struct{}, // Signal completion on this channel
	errResult *error, // Pointer to store the final error result
	stream pb.FileService_UploadServer, // Stream to send the final response
) {
	defer close(done) // Ensure done channel is closed when this goroutine exits

	// Call the use case, passing the pipe reader.
	// Assuming UploadFile reads pr until EOF or error.
	// Pass 0 for size, as the actual size isn't known until the *other* goroutine finishes.
	// The use case should ideally not rely on this size parameter for reading the stream.
	fileEntity, err := s.uc.UploadFile(ctx, filename, pr, 0, mimeType, uploaderID)
	if err != nil {
		// s.uc.GetLogger().Error("runUpload: Use case UploadFile failed", "error", err)
		pr.CloseWithError(err)                                // Ensure reader knows about the error if writer is still active
		*errResult = coreController.MapErrorToHttpStatus(err) // Map domain error to gRPC status
		return
	}
	// If use case succeeded, close the reader part (no more data needed)
	pr.Close()

	// s.uc.GetLogger().Info("runUpload: Use case UploadFile succeeded", "fileId", fileEntity.ID)

	// Map result entity back to proto
	fileProto, mapErr := s.mapper.FileEntityToProto(fileEntity)
	if mapErr != nil {
		// s.uc.GetLogger().Error("runUpload: Failed to map result entity to proto", "error", mapErr)
		*errResult = status.Errorf(codes.Internal, "failed to map result: %v", mapErr)
		return
	}

	// Send the final response back to the client and close the stream
	if sendErr := stream.SendAndClose(&pb.UploadFileResponse{File: fileProto}); sendErr != nil {
		// s.uc.GetLogger().Error("runUpload: Failed to send final response", "error", sendErr)
		*errResult = status.Errorf(codes.Internal, "failed to send upload response: %v", sendErr)
		return
	}

	// s.uc.GetLogger().Info("runUpload: Successfully sent response and closed stream")
	*errResult = nil // Explicitly signal success
}

// Delete implements proto.FileServiceServer.
func (s *fileServer) Delete(ctx context.Context, req *pb.DeleteFileRequest) (*emptypb.Empty, error) {
	id, err := uuid.Parse(req.GetId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid file ID format: %v", err)
	}
	hardDelete := req.GetHardDelete()

	// TODO: Add authorization check - can the user delete this file?
	// claims := middleware.GetClaimsFromContext(ctx)
	// if claims == nil { return nil, status.Errorf(codes.Unauthenticated, "missing claims") }
	// Fetch file entity to check uploaderID against claims.Subject?

	err = s.uc.DeleteFile(ctx, id, hardDelete) // Use DeleteFile explicitly
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}
	return &emptypb.Empty{}, nil
}

// ListUserFiles implements proto.FileServiceServer.
func (s *fileServer) ListUserFiles(ctx context.Context, req *pb.ListUserFilesRequest) (*pb.ListUserFilesResponse, error) {
	userID, err := uuid.Parse(req.GetUserId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid user ID format: %v", err)
	}

	opts := s.mapper.ProtoListRequestToFilterOptions(req.Options) // Map options

	result, err := s.uc.ListUserFiles(ctx, userID, opts)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// Map pagination result to proto list response
	response, err := s.mapper.FilePaginationResultToProtoList(result)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to map result list: %v", err)
	}

	return response, nil
}

// GetByIDs handles requests to retrieve multiple files by their IDs.
func (s *fileServer) GetByIDs(ctx context.Context, req *pb.GetFilesByIDsRequest) (*pb.GetFilesByIDsResponse, error) {
	rawIDs := req.GetIds()
	if len(rawIDs) == 0 {
		// Return empty response, not an error, if no IDs requested
		return &pb.GetFilesByIDsResponse{Files: []*pb.File{}}, nil
	}

	ids := make([]uuid.UUID, 0, len(rawIDs))
	for _, idStr := range rawIDs {
		id, err := uuid.Parse(idStr)
		if err != nil {
			// Log this? Return specific error?
			return nil, status.Errorf(codes.InvalidArgument, "invalid file ID format '%s': %v", idStr, err)
		}
		ids = append(ids, id)
	}

	// Call the use case method
	fileEntities, err := s.uc.GetByIDs(ctx, ids)
	if err != nil {
		// Map potential use case errors (like internal DB errors)
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// Map the resulting entities to proto messages
	// Need a mapper function for this (e.g., FileListToProto)
	filesProto, err := s.mapper.FileListToProto(fileEntities) // Assuming this method exists
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to map file list results: %v", err)
	}

	return &pb.GetFilesByIDsResponse{
		Files: filesProto,
	}, nil
}
