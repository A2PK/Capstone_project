package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure" // TODO: Use secure credentials
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	userPb "golang-microservices-boilerplate/proto/user-service" // Adjust import path if needed
	"golang-microservices-boilerplate/services/api-gateway/internal/domain"
)

// Constants specific to FileService uploads (can reuse from binaryFileHandler if identical)
// const (
// 	uploadTimeout = 10 * time.Minute
// 	maxUploadSize = 1 << 30 // 1GB
// 	chunkSize     = 1 << 20 // 1MB
// )

const (
	// maxUploadSize defines the maximum size for file uploads.
	uploadTimeout = 10 * time.Minute
	maxUploadSize = 2 << 30  // 2GB
	chunkSize     = 32 << 20 // 32MB
)

// handleFileServiceUpload returns the custom HTTP handler function for FileService uploads.
// It handles multipart form data containing 'name', 'type', and 'file'.
func handleFileServiceUpload(userServiceAddr string) runtime.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request, pathParams map[string]string) {
		// 1. Parse Multipart Form
		// Note: Ensure maxUploadSize is defined or imported if needed globally
		if err := r.ParseMultipartForm(maxUploadSize); err != nil {
			http.Error(w, fmt.Sprintf("failed to parse multipart form: %v", err), http.StatusBadRequest)
			return
		}

		// 2. Extract Metadata Fields (name, type)
		fileName := r.FormValue("name")
		if fileName == "" {
			http.Error(w, "form field 'name' is required", http.StatusBadRequest)
			return
		}

		fileType := r.FormValue("type")
		if fileType == "" {
			http.Error(w, "form field 'type' is required", http.StatusBadRequest)
			return
		}

		// 3. Extract File Field
		file, _, err := r.FormFile("file")
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to get form file 'file': %v", err), http.StatusBadRequest)
			return
		}
		defer file.Close()

		// --- Start gRPC Streaming ---

		// Use request context with timeout
		// Note: Ensure uploadTimeout is defined or imported
		ctx, cancel := context.WithTimeout(r.Context(), uploadTimeout)
		defer cancel()

		// 4. Establish gRPC Client Connection to User Service
		opts := []grpc.DialOption{
			grpc.WithTransportCredentials(insecure.NewCredentials()), // FIXME: Use secure credentials!
			grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(maxUploadSize), grpc.MaxCallSendMsgSize(maxUploadSize)),
		}
		conn, err := grpc.NewClient(userServiceAddr, opts...)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to connect to user service (%s): %v", userServiceAddr, err), http.StatusInternalServerError)
			return
		}
		defer conn.Close()

		client := userPb.NewFileServiceClient(conn)

		// add metadata authorization token if needed
		md := metadata.MD{}
		if authHeader := r.Header.Get("Authorization"); authHeader != "" {
			md.Set("authorization", authHeader)
		}
		ctx = metadata.NewOutgoingContext(ctx, md)

		// 5. Call Streaming RPC: FileService.Upload
		stream, err := client.Upload(ctx)
		if err != nil {
			st, _ := status.FromError(err)
			http.Error(w, fmt.Sprintf("failed to start upload stream: %s", st.Message()), grpcStatusToHTTP(st.Code()))
			return
		}

		// 6. Send Metadata Messages (Name first, then Type)
		if err := stream.Send(&userPb.UploadFileRequest{RequestData: &userPb.UploadFileRequest_Name{Name: fileName}}); err != nil {
			st, _ := status.FromError(err)
			http.Error(w, fmt.Sprintf("failed to send filename: %s", st.Message()), grpcStatusToHTTP(st.Code()))
			return
		}
		if err := stream.Send(&userPb.UploadFileRequest{RequestData: &userPb.UploadFileRequest_Type{Type: fileType}}); err != nil {
			st, _ := status.FromError(err)
			http.Error(w, fmt.Sprintf("failed to send filetype: %s", st.Message()), grpcStatusToHTTP(st.Code()))
			return
		}

		// 7. Stream File Content (ChunkData)
		// Note: Ensure chunkSize is defined or imported
		buffer := make([]byte, chunkSize)
		for {
			n, readErr := file.Read(buffer)
			if n > 0 {
				// Send chunk data
				if sendErr := stream.Send(&userPb.UploadFileRequest{RequestData: &userPb.UploadFileRequest_ChunkData{ChunkData: buffer[:n]}}); sendErr != nil {
					st, _ := status.FromError(sendErr)
					// Check for graceful closure or specific errors
					if sendErr == io.EOF || st.Code() == codes.Canceled || st.Code() == codes.Unavailable {
						fmt.Printf("INFO (FileService Upload): Send encountered %v, proceeding to CloseAndRecv\n", sendErr)
						break // Exit loop and attempt CloseAndRecv
					}
					http.Error(w, fmt.Sprintf("failed to send data chunk: %s", st.Message()), grpcStatusToHTTP(st.Code()))
					return
				}
			}
			if readErr == io.EOF {
				break // End of file
			}
			if readErr != nil {
				http.Error(w, fmt.Sprintf("error reading file chunk: %v", readErr), http.StatusInternalServerError)
				return
			}
		}

		// 8. Close Stream and Get Response (UploadFileResponse)
		resp, err := stream.CloseAndRecv() // resp is *userPb.UploadFileResponse
		if err != nil {
			if err == io.EOF {
				http.Error(w, "upload failed: server closed connection unexpectedly", http.StatusServiceUnavailable)
			} else {
				st, ok := status.FromError(err)
				if ok {
					http.Error(w, fmt.Sprintf("upload processing failed: %s", st.Message()), grpcStatusToHTTP(st.Code()))
				} else {
					http.Error(w, fmt.Sprintf("upload failed with unexpected error: %v", err), http.StatusInternalServerError)
				}
			}
			return
		}

		// --- Send Final HTTP Response (UploadFileResponse JSON) ---
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK) // 200 OK
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			fmt.Printf("ERROR: Failed to encode FileService upload response to client: %v\n", err)
		}
	}
}

// registerFileServiceCustomHandlers registers custom handlers specific to the FileService.
func registerFileServiceCustomHandlers(mux *runtime.ServeMux, service domain.Service) error {
	if service.Endpoint == "" {
		return fmt.Errorf("cannot register custom handlers: endpoint missing for service %s", service.Name)
	}

	uploadPath := "/api/v1/files" // The target API path for file uploads

	// Register the custom handler for the upload path
	err := mux.HandlePath("POST", uploadPath, handleFileServiceUpload(service.Endpoint))
	if err != nil {
		return fmt.Errorf("failed to register custom handler for path %s on service %s: %w", uploadPath, service.Name, err)
	}
	return nil
}

// Note: grpcStatusToHTTP function needs to be accessible, either defined here,
// in binaryFileHandler.go (and imported), or moved to a common utility package.
// For simplicity, assume it's accessible or duplicate it if needed.
// Also assumes constants like maxUploadSize, chunkSize, uploadTimeout are defined/accessible.

// grpcStatusToHTTP maps gRPC status codes to HTTP status codes.
func grpcStatusToHTTP(code codes.Code) int {
	switch code {
	case codes.OK:
		return http.StatusOK
	case codes.Canceled:
		return http.StatusRequestTimeout // Or 499 Client Closed Request if using Nginx/specific proxies
	case codes.Unknown:
		return http.StatusInternalServerError
	case codes.InvalidArgument:
		return http.StatusBadRequest
	case codes.DeadlineExceeded:
		return http.StatusGatewayTimeout
	case codes.NotFound:
		return http.StatusNotFound
	case codes.AlreadyExists:
		return http.StatusConflict
	case codes.PermissionDenied:
		return http.StatusForbidden
	case codes.ResourceExhausted:
		return http.StatusTooManyRequests
	case codes.FailedPrecondition:
		return http.StatusBadRequest
	case codes.Aborted:
		return http.StatusConflict
	case codes.OutOfRange:
		return http.StatusBadRequest
	case codes.Unimplemented:
		return http.StatusNotImplemented
	case codes.Internal:
		return http.StatusInternalServerError
	case codes.Unavailable:
		return http.StatusServiceUnavailable
	case codes.DataLoss:
		return http.StatusInternalServerError
	case codes.Unauthenticated:
		return http.StatusUnauthorized
	default:
		return http.StatusInternalServerError
	}
}
