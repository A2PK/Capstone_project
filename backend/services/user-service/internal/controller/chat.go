package controller

import (
	"context"
	"net/http"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"

	coreController "golang-microservices-boilerplate/pkg/core/controller"
	"golang-microservices-boilerplate/pkg/middleware"

	// coreTypes "golang-microservices-boilerplate/pkg/core/types" // Import if needed
	corePb "golang-microservices-boilerplate/proto/core" // Import core proto for PaginationInfo
	pb "golang-microservices-boilerplate/proto/user-service"
	"golang-microservices-boilerplate/services/user-service/internal/usecase"
)

// Ensure chatServer implements pb.DirectMessageServiceServer.
var _ pb.DirectMessageServiceServer = (*chatServer)(nil)

type chatServer struct {
	pb.UnimplementedDirectMessageServiceServer
	uc     usecase.ChatUsecase
	mapper Mapper // Use the general Mapper
}

// NewChatServer creates a new gRPC chat server instance.
func NewChatServer(uc usecase.ChatUsecase, mapper Mapper) pb.DirectMessageServiceServer {
	return &chatServer{
		uc:     uc,
		mapper: mapper,
	}
}

// RegisterDirectMessageServiceServer registers the chat service implementation.
func RegisterDirectMessageServiceServer(s *grpc.Server, uc usecase.ChatUsecase, mapper Mapper) {
	server := NewChatServer(uc, mapper)
	pb.RegisterDirectMessageServiceServer(s, server)
}

// --- Implement DirectMessageServiceServer Methods ---

func (s *chatServer) SendMessage(ctx context.Context, req *pb.SendMessageRequest) (*pb.SendMessageResponse, error) {
	// Get sender ID from context (REQUIRED)
	senderID, err := middleware.GetUserIdFromGRPCContext(ctx)
	if err != nil {
		return nil, status.Errorf(http.StatusUnauthorized, "%v", err)
	}

	receiverID, err := uuid.Parse(req.GetReceiverId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid receiver ID format: %v", err)
	}

	messageContent := req.GetMessage()
	if messageContent == "" {
		return nil, status.Errorf(codes.InvalidArgument, "message content cannot be empty")
	}

	// Call use case
	chatEntity, err := s.uc.SendMessage(ctx, senderID, receiverID, messageContent)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// Map result entity back to proto
	chatProto, err := s.mapper.ChatMessageEntityToProto(chatEntity)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to map result: %v", err)
	}

	return &pb.SendMessageResponse{Message: chatProto}, nil
}

func (s *chatServer) ListMessagesBetweenUsers(ctx context.Context, req *pb.ListMessagesBetweenUsersRequest) (*pb.ListMessagesBetweenUsersResponse, error) {
	userID1, err := uuid.Parse(req.GetUserId1())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid user_id1 format: %v", err)
	}
	userID2, err := uuid.Parse(req.GetUserId2())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid user_id2 format: %v", err)
	}

	// TODO: Authorization Check: Can the current user view messages between userID1 and userID2?

	opts := s.mapper.ProtoListRequestToFilterOptions(req.Options)

	result, err := s.uc.ListMessagesBetweenUsers(ctx, userID1, userID2, opts)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// Map result entity back to proto
	messagesProto := make([]*pb.DirectChatMessage, 0, len(result.Items))
	for _, msgEntity := range result.Items {
		msgProto, mapErr := s.mapper.ChatMessageEntityToProto(msgEntity)
		if mapErr != nil {
			// Log or handle error?
			return nil, status.Errorf(codes.Internal, "failed to map message %s: %v", msgEntity.ID, mapErr)
		}
		messagesProto = append(messagesProto, msgProto)
	}

	// Manually construct PaginationInfo from the result
	paginationInfo := &corePb.PaginationInfo{
		TotalItems: result.TotalItems,
		Limit:      int32(result.Limit),
		Offset:     int32(result.Offset),
	}

	return &pb.ListMessagesBetweenUsersResponse{
		Messages:       messagesProto,
		PaginationInfo: paginationInfo,
	}, nil
}

func (s *chatServer) ListUnseenMessagesForUser(ctx context.Context, req *pb.ListUnseenMessagesForUserRequest) (*pb.ListUnseenMessagesForUserResponse, error) {
	// Get user ID from context (REQUIRED)
	userID, err := middleware.GetUserIdFromGRPCContext(ctx)
	if err != nil {
		return nil, status.Errorf(http.StatusUnauthorized, "%v", err)
	}

	messages, err := s.uc.ListUnseenMessagesForUser(ctx, userID)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	messagesProto := make([]*pb.DirectChatMessage, 0, len(messages))
	for _, msgEntity := range messages {
		msgProto, mapErr := s.mapper.ChatMessageEntityToProto(msgEntity)
		if mapErr != nil {
			return nil, status.Errorf(codes.Internal, "failed to map message %s: %v", msgEntity.ID, mapErr)
		}
		messagesProto = append(messagesProto, msgProto)
	}

	return &pb.ListUnseenMessagesForUserResponse{Messages: messagesProto}, nil
}

func (s *chatServer) MarkMessagesAsRead(ctx context.Context, req *pb.MarkMessagesAsReadRequest) (*emptypb.Empty, error) {
	// Get user ID from context (REQUIRED) - this is the reader
	readerID, er := middleware.GetUserIdFromGRPCContext(ctx)
	if er != nil {
		return nil, status.Errorf(http.StatusUnauthorized, "%v", er)
	}
	if len(req.GetMessageIds()) == 0 {
		return nil, status.Errorf(codes.InvalidArgument, "message_ids cannot be empty")
	}

	messageIDs := make([]uuid.UUID, 0, len(req.GetMessageIds()))
	for _, idStr := range req.GetMessageIds() {
		id, err := uuid.Parse(idStr)
		if err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "invalid message ID format '%s': %v", idStr, err)
		}
		messageIDs = append(messageIDs, id)
	}

	err := s.uc.MarkMessagesAsRead(ctx, messageIDs, readerID)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	return &emptypb.Empty{}, nil
}
