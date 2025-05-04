package controller

import (
	"context"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"

	coreController "golang-microservices-boilerplate/pkg/core/controller" // Import core proto
	pb "golang-microservices-boilerplate/proto/user-service"
	"golang-microservices-boilerplate/services/user-service/internal/usecase"
)

// Ensure requestServer implements pb.RequestServiceServer.
var _ pb.RequestServiceServer = (*requestServer)(nil)

type requestServer struct {
	pb.UnimplementedRequestServiceServer
	uc     usecase.RequestUsecase
	mapper Mapper // Use the general Mapper
}

// NewRequestServer creates a new gRPC request server instance.
func NewRequestServer(uc usecase.RequestUsecase, mapper Mapper) pb.RequestServiceServer {
	return &requestServer{
		uc:     uc,
		mapper: mapper,
	}
}

// RegisterRequestServiceServer registers the request service implementation.
func RegisterRequestServiceServer(s *grpc.Server, uc usecase.RequestUsecase, mapper Mapper) {
	server := NewRequestServer(uc, mapper)
	pb.RegisterRequestServiceServer(s, server)
}

// --- Implement RequestServiceServer Methods ---

func (s *requestServer) Create(ctx context.Context, req *pb.CreateRequestRequest) (*pb.CreateRequestResponse, error) {
	// Map proto request to entity
	requestEntity, err := s.mapper.RequestProtoCreateToEntity(req)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "failed to map request: %v", err)
	}

	// TODO: Validate SenderID matches authenticated user?

	requestEntity, err = s.uc.CreateRequestWithFiles(ctx, requestEntity, req.GetFileIds())
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// Map result entity back to proto
	requestProto, err := s.mapper.RequestEntityToProto(requestEntity)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to map result: %v", err)
	}

	return &pb.CreateRequestResponse{Request: requestProto}, nil
}

func (s *requestServer) GetByID(ctx context.Context, req *pb.GetRequestByIDRequest) (*pb.GetRequestByIDResponse, error) {
	id, err := uuid.Parse(req.GetId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid request ID format: %v", err)
	}

	requestEntity, err := s.uc.GetByID(ctx, id) // Assuming base GetByID method
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// TODO: Authorization Check: Can the current user view this request? (check sender/receiver ID)

	requestProto, err := s.mapper.RequestEntityToProto(requestEntity)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to map result: %v", err)
	}

	return &pb.GetRequestByIDResponse{Request: requestProto}, nil
}

func (s *requestServer) List(ctx context.Context, req *pb.ListRequestsRequest) (*pb.ListRequestsResponse, error) {
	opts := s.mapper.ProtoListRequestToFilterOptions(req.Options)

	// TODO: Authorization Check? Maybe list only requests involving the current user?
	// Requires specific use case method like FindByUserParticipant or filtering here.

	result, err := s.uc.List(ctx, opts) // Assuming base List handles filters
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	response, err := s.mapper.RequestPaginationResultToProtoList(result)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to map result list: %v", err)
	}

	return response, nil
}

func (s *requestServer) Update(ctx context.Context, req *pb.UpdateRequestRequest) (*pb.UpdateRequestResponse, error) {
	id, err := uuid.Parse(req.GetId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid request ID format: %v", err)
	}

	// 1. Get existing entity
	existingRequest, err := s.uc.GetByID(ctx, id) // Assuming base GetByID
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// TODO: Authorization Check: Can the current user update this request? (check sender/receiver ID, status?)

	// 2. Apply updates from proto request
	if err := s.mapper.RequestApplyProtoUpdateToEntity(req, existingRequest); err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "failed to map update request: %v", err)
	}

	// 3. Call use case Update
	err = s.uc.Update(ctx, existingRequest) // Assuming base Update
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// 4. Map updated entity back to proto
	requestProto, err := s.mapper.RequestEntityToProto(existingRequest)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to map result: %v", err)
	}

	return &pb.UpdateRequestResponse{Request: requestProto}, nil
}

func (s *requestServer) Delete(ctx context.Context, req *pb.DeleteRequestRequest) (*emptypb.Empty, error) {
	id, err := uuid.Parse(req.GetId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid request ID format: %v", err)
	}
	hardDelete := req.GetHardDelete()

	// TODO: Authorization Check: Can the current user delete this request?

	err = s.uc.Delete(ctx, id, hardDelete) // Assuming base Delete
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	return &emptypb.Empty{}, nil
}

func (s *requestServer) FindByUserParticipant(ctx context.Context, req *pb.FindByUserParticipantRequest) (*pb.FindByUserParticipantResponse, error) {
	userID, err := uuid.Parse(req.GetUserId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid user_id format: %v", err)
	}

	// TODO: Authorization check - can the current user list requests for req.GetUserId()?

	opts := s.mapper.ProtoListRequestToFilterOptions(req.Options)

	result, err := s.uc.FindByUserParticipant(ctx, userID, req.Status, opts)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// Map result
	response, err := s.mapper.RequestPaginationResultToProtoList(result) // Use existing list response mapper
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to map result list: %v", err)
	}

	// Need to wrap the result in FindByUserParticipantResponse
	finalResponse := &pb.FindByUserParticipantResponse{
		Requests:       response.Requests,
		PaginationInfo: response.PaginationInfo,
	}

	return finalResponse, nil
}
