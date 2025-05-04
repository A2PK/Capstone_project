package controller

import (
	"context"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"

	coreController "golang-microservices-boilerplate/pkg/core/controller"
	"golang-microservices-boilerplate/pkg/middleware"

	coreTypes "golang-microservices-boilerplate/pkg/core/types"
	corePb "golang-microservices-boilerplate/proto/core"
	pb "golang-microservices-boilerplate/proto/user-service"
	"golang-microservices-boilerplate/services/user-service/internal/entity"
	"golang-microservices-boilerplate/services/user-service/internal/usecase"
)

// Ensure notificationServer implements pb.NotificationServiceServer.
var _ pb.NotificationServiceServer = (*notificationServer)(nil)

type notificationServer struct {
	pb.UnimplementedNotificationServiceServer
	uc     usecase.NotificationUsecase
	mapper Mapper // Use the general Mapper
}

// NewNotificationServer creates a new gRPC notification server instance.
func NewNotificationServer(uc usecase.NotificationUsecase, mapper Mapper) pb.NotificationServiceServer {
	return &notificationServer{
		uc:     uc,
		mapper: mapper,
	}
}

// RegisterNotificationServiceServer registers the notification service implementation.
func RegisterNotificationServiceServer(s *grpc.Server, uc usecase.NotificationUsecase, mapper Mapper) {
	server := NewNotificationServer(uc, mapper)
	pb.RegisterNotificationServiceServer(s, server)
}

// --- Implement NotificationServiceServer Methods ---

func (s *notificationServer) Create(ctx context.Context, req *pb.CreateNotificationRequest) (*pb.CreateNotificationResponse, error) {
	// Map proto request to entity
	notificationEntity, err := s.mapper.NotificationProtoCreateToEntity(req)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "failed to map request: %v", err)
	}

	// Call use case
	err = s.uc.Create(ctx, notificationEntity) // Assuming base Create method
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// Map result entity back to proto
	notificationProto, err := s.mapper.NotificationEntityToProto(notificationEntity)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to map result: %v", err)
	}

	return &pb.CreateNotificationResponse{Notification: notificationProto}, nil
}

func (s *notificationServer) GetByID(ctx context.Context, req *pb.GetNotificationByIDRequest) (*pb.GetNotificationByIDResponse, error) {
	id, err := uuid.Parse(req.GetId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid notification ID format: %v", err)
	}

	notificationEntity, err := s.uc.GetByID(ctx, id) // Assuming base GetByID method
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	notificationProto, err := s.mapper.NotificationEntityToProto(notificationEntity)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to map result: %v", err)
	}

	return &pb.GetNotificationByIDResponse{Notification: notificationProto}, nil
}

// FindByUser retrieves notifications for a specific user with optional filter by read status
func (s *notificationServer) FindByUser(ctx context.Context, req *pb.FindByUserRequest) (*pb.FindByUserResponse, error) {
	userID, err := uuid.Parse(req.GetUserId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid user ID format: %v", err)
	}

	// Use FilterOptions directly
	var opts coreTypes.FilterOptions
	if req.Options != nil {
		opts.Limit = int(req.Options.GetLimit())
		opts.Offset = int(req.Options.GetOffset())
		opts.SortBy = req.Options.GetSortBy()
		opts.SortDesc = req.Options.GetSortDesc()
		opts.Filters = make(map[string]interface{})
		for k, v := range req.Options.GetFilters() {
			opts.Filters[k] = v
		}
	}

	// Get optional read status filter
	var readStatus *bool
	if req.Read != nil {
		read := req.GetRead()
		readStatus = &read
	}

	// Call the usecase with the proper parameters
	result, err := s.uc.ListNotificationsForUser(ctx, userID, readStatus, opts)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// Map result to proto response
	notifications := make([]*pb.Notification, 0, len(result.Items))
	for _, n := range result.Items {
		notificationProto, err := s.mapper.NotificationEntityToProto(n)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to map notification: %v", err)
		}
		notifications = append(notifications, notificationProto)
	}

	// Create response with pagination info
	response := &pb.FindByUserResponse{
		Notifications: notifications,
		PaginationInfo: &corePb.PaginationInfo{
			TotalItems: result.TotalItems,
			Limit:      int32(result.Limit),
			Offset:     int32(result.Offset),
		},
	}

	return response, nil
}

func (s *notificationServer) List(ctx context.Context, req *pb.ListNotificationsRequest) (*pb.ListNotificationsResponse, error) {
	opts := s.mapper.ProtoListRequestToFilterOptions(req.Options)

	// Get user ID from context (REQUIRED)
	userID, err := middleware.GetUserIdFromGRPCContext(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Unauthenticated, "%v", err)
	}

	opts.Filters["user_id"] = userID // Assuming filters are a map[string]string

	result, err := s.uc.List(ctx, opts) // Assuming base List handles filters
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	response, err := s.mapper.NotificationPaginationResultToProtoList(result)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to map result list: %v", err)
	}

	return response, nil
}

func (s *notificationServer) Update(ctx context.Context, req *pb.UpdateNotificationRequest) (*pb.UpdateNotificationResponse, error) {
	id, err := uuid.Parse(req.GetId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid notification ID format: %v", err)
	}

	// 1. Get existing entity
	existingNotification, err := s.uc.GetByID(ctx, id) // Assuming base GetByID
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// TODO: Authorization Check: Can the current user update this notification? (e.g., check UserID)

	// 2. Apply updates from proto request
	if err := s.mapper.NotificationApplyProtoUpdateToEntity(req, existingNotification); err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "failed to map update request: %v", err)
	}

	// 3. Call use case Update
	err = s.uc.Update(ctx, existingNotification) // Assuming base Update
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// 4. Map updated entity back to proto
	notificationProto, err := s.mapper.NotificationEntityToProto(existingNotification)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to map result: %v", err)
	}

	return &pb.UpdateNotificationResponse{Notification: notificationProto}, nil
}

func (s *notificationServer) Delete(ctx context.Context, req *pb.DeleteNotificationRequest) (*emptypb.Empty, error) {
	id, err := uuid.Parse(req.GetId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid notification ID format: %v", err)
	}
	hardDelete := req.GetHardDelete()

	// TODO: Authorization Check: Can the current user delete this notification?

	err = s.uc.Delete(ctx, id, hardDelete) // Assuming base Delete
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	return &emptypb.Empty{}, nil
}

func (s *notificationServer) MarkAsRead(ctx context.Context, req *pb.MarkNotificationsAsReadRequest) (*emptypb.Empty, error) {
	if len(req.GetNotificationIds()) == 0 {
		return nil, status.Errorf(codes.InvalidArgument, "notification_ids cannot be empty")
	}

	notificationIDs := make([]uuid.UUID, 0, len(req.GetNotificationIds()))
	for _, idStr := range req.GetNotificationIds() {
		id, err := uuid.Parse(idStr)
		if err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "invalid notification ID format '%s': %v", idStr, err)
		}
		notificationIDs = append(notificationIDs, id)
	}

	// Get user ID from context (REQUIRED)
	userID, err := middleware.GetUserIdFromGRPCContext(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Unauthenticated, "%v", err)
	}
	// Call use case method
	err = s.uc.MarkNotificationsAsRead(ctx, notificationIDs, userID) // Pass user ID for authorization
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	return &emptypb.Empty{}, nil
}

// CreateMany handles bulk creation of notifications.
func (s *notificationServer) CreateMany(ctx context.Context, req *pb.CreateNotificationsRequest) (*pb.CreateNotificationsResponse, error) {
	notificationsToCreate := req.GetNotifications()
	if len(notificationsToCreate) == 0 {
		return nil, status.Errorf(codes.InvalidArgument, "notification list cannot be empty")
	}

	entityList := make([]*entity.Notification, 0, len(notificationsToCreate))
	for i, protoReq := range notificationsToCreate {
		entity, err := s.mapper.NotificationProtoCreateToEntity(protoReq)
		if err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "failed to map notification at index %d: %v", i, err)
		}
		entityList = append(entityList, entity)
	}

	// Call the use case to create entities in bulk
	createdEntities, err := s.uc.CreateMany(ctx, entityList)
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err) // Map use case error
	}

	// Map the created entities back to proto for the response
	protoList := make([]*pb.Notification, 0, len(createdEntities))
	for i, createdEntity := range createdEntities {
		protoNotification, err := s.mapper.NotificationEntityToProto(createdEntity)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to map created notification at index %d: %v", i, err)
		}
		protoList = append(protoList, protoNotification)
	}

	return &pb.CreateNotificationsResponse{
		Notifications: protoList,
	}, nil
}
