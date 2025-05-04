package controller

import (
	"errors"
	"fmt"

	"github.com/google/uuid"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	coreTypes "golang-microservices-boilerplate/pkg/core/types"
	corePb "golang-microservices-boilerplate/proto/core"
	pb "golang-microservices-boilerplate/proto/user-service"
	"golang-microservices-boilerplate/services/user-service/internal/entity"
	userschema "golang-microservices-boilerplate/services/user-service/internal/schema"
	// Import usecase package for ConversationSummary struct
	// Keep usecase import only if needed, maybe not needed now
	// userservice_usecase "golang-microservices-boilerplate/services/user-service/internal/usecase"
)

// Mapper defines the interface for mapping between gRPC proto messages and internal types for ALL services.
type Mapper interface {
	// User Mappings
	UserEntityToProto(user *entity.User) (*pb.User, error)
	UserProtoCreateToEntity(req *pb.CreateUserRequest) (*entity.User, error)
	UserApplyProtoUpdateToEntity(req *pb.UpdateUserRequest, existingUser *entity.User) error
	UserProtoLoginToSchema(req *pb.LoginRequest) (userschema.LoginCredentials, error)
	UserSchemaLoginResultToProto(result *userschema.LoginResult) (*pb.LoginResponse, error)
	UserSchemaRefreshResultToProto(result *userschema.RefreshResult) (*pb.RefreshResponse, error)
	UserPaginationResultToProtoList(result *coreTypes.PaginationResult[entity.User]) (*pb.ListUsersResponse, error)

	// File Mappings (Placeholders - Implement details)
	FileEntityToProto(file *entity.File) (*pb.File, error)
	FilePaginationResultToProtoList(result *coreTypes.PaginationResult[entity.File]) (*pb.ListUserFilesResponse, error) // Assuming response type for ListUserFiles
	FileListToProto(files []*entity.File) ([]*pb.File, error)

	// Article Mappings (Placeholders)
	ArticleEntityToProto(article *entity.Article) (*pb.Article, error)
	ArticleProtoCreateToEntity(req *pb.CreateArticleRequest) (*entity.Article, error)
	ArticleApplyProtoUpdateToEntity(req *pb.UpdateArticleRequest, existingArticle *entity.Article) error
	ArticlePaginationResultToProtoList(result *coreTypes.PaginationResult[entity.Article]) (*pb.ListArticlesResponse, error)

	// Request Mappings (Placeholders)
	RequestEntityToProto(request *entity.Request) (*pb.Request, error)
	RequestProtoCreateToEntity(req *pb.CreateRequestRequest) (*entity.Request, error)
	RequestApplyProtoUpdateToEntity(req *pb.UpdateRequestRequest, existingRequest *entity.Request) error
	RequestPaginationResultToProtoList(result *coreTypes.PaginationResult[entity.Request]) (*pb.ListRequestsResponse, error) // Adjust response type if needed

	// Chat Mappings
	ChatMessageEntityToProto(msg *entity.ChatMessage) (*pb.DirectChatMessage, error)

	// Notification Mappings (Placeholders)
	NotificationEntityToProto(notification *entity.Notification) (*pb.Notification, error)
	NotificationProtoCreateToEntity(req *pb.CreateNotificationRequest) (*entity.Notification, error)
	NotificationApplyProtoUpdateToEntity(req *pb.UpdateNotificationRequest, existingNotification *entity.Notification) error
	NotificationPaginationResultToProtoList(result *coreTypes.PaginationResult[entity.Notification]) (*pb.ListNotificationsResponse, error)

	// Common Mappings
	ProtoListRequestToFilterOptions(options *corePb.FilterOptions) coreTypes.FilterOptions // Use corePb.FilterOptions

}

// Ensure mapperImpl implements Mapper interface.
var _ Mapper = (*mapperImpl)(nil)

// mapperImpl handles mapping between gRPC proto messages and internal types.
type mapperImpl struct{}

// NewMapper creates a new instance of the general mapper.
func NewMapper() Mapper {
	return &mapperImpl{}
}

// --- User Mappings ---

// UserEntityToProto converts an entity.User to a proto.User.
func (m *mapperImpl) UserEntityToProto(user *entity.User) (*pb.User, error) {
	if user == nil {
		return nil, errors.New("cannot map nil user entity to proto")
	}
	var deletedAt *timestamppb.Timestamp
	if user.DeletedAt != nil {
		deletedAt = timestamppb.New(*user.DeletedAt)
	}
	var lastLoginAt *timestamppb.Timestamp
	if user.LastLoginAt != nil {
		lastLoginAt = timestamppb.New(*user.LastLoginAt)
	}

	return &pb.User{
		Id:          user.ID.String(),
		Username:    user.Username,
		Email:       user.Email,
		FirstName:   user.FirstName,
		LastName:    user.LastName,
		Role:        string(user.Role),
		IsActive:    user.IsActive,
		CreatedAt:   timestamppb.New(user.CreatedAt),
		UpdatedAt:   timestamppb.New(user.UpdatedAt),
		DeletedAt:   deletedAt,
		LastLoginAt: lastLoginAt,
		Phone:       user.Phone,
		Address:     user.Address,
		Age:         user.Age,
		ProfilePic:  user.ProfilePic,
	}, nil
}

// UserProtoCreateToEntity converts a proto.CreateUserRequest directly to an entity.User pointer.
func (m *mapperImpl) UserProtoCreateToEntity(req *pb.CreateUserRequest) (*entity.User, error) {
	if req == nil {
		return nil, errors.New("cannot map nil create user request to entity")
	}
	user := &entity.User{
		Username:  req.Username,
		Email:     req.Email,
		Password:  req.Password, // Will be hashed by BeforeCreate hook
		FirstName: req.FirstName,
		LastName:  req.LastName,
		Role:      entity.Role(req.Role), // Default applied in BeforeCreate if invalid/empty
		// Safely dereference optional string fields
		Phone:      derefString(req.Phone),
		Address:    derefString(req.Address),
		Age:        req.GetAge(), // Use GetAge() for nil safety
		ProfilePic: derefString(req.ProfilePic),
		IsActive:   false, // Explicitly set default, though BeforeCreate/DB default handles it
	}
	if !user.Role.IsValid() {
		return nil, fmt.Errorf("invalid role provided: %s", req.Role)
	}
	if user.Email == "" || user.FirstName == "" || user.LastName == "" {
		return nil, errors.New("email, first name, and last name are required")
	}

	return user, nil
}

// UserApplyProtoUpdateToEntity applies fields from proto.UpdateUserRequest to an existing entity.User.
func (m *mapperImpl) UserApplyProtoUpdateToEntity(req *pb.UpdateUserRequest, existingUser *entity.User) error {
	if req == nil || existingUser == nil {
		return errors.New("update user request and existing entity must not be nil")
	}

	if req.Username != nil {
		existingUser.Username = req.Username.Value
	}
	if req.Email != nil {
		existingUser.Email = req.Email.Value
	}
	if req.Password != nil {
		existingUser.Password = req.Password.Value
	}
	if req.FirstName != nil {
		existingUser.FirstName = req.FirstName.Value
	}
	if req.LastName != nil {
		existingUser.LastName = req.LastName.Value
	}
	if req.Role != nil {
		role := entity.Role(req.Role.Value)
		if role.IsValid() {
			existingUser.Role = role
		} else {
			return fmt.Errorf("invalid role provided for update: %s", req.Role.Value)
		}
	}
	if req.IsActive != nil {
		existingUser.IsActive = req.IsActive.Value
	}
	if req.Phone != nil {
		existingUser.Phone = req.Phone.Value
	}
	if req.Address != nil {
		existingUser.Address = req.Address.Value
	}
	if req.Age != nil {
		existingUser.Age = req.Age.Value
	}
	if req.ProfilePic != nil {
		existingUser.ProfilePic = req.ProfilePic.Value
	}

	return nil
}

// UserProtoLoginToSchema converts proto.LoginRequest to schema.LoginCredentials.
func (m *mapperImpl) UserProtoLoginToSchema(req *pb.LoginRequest) (userschema.LoginCredentials, error) {
	if req == nil {
		return userschema.LoginCredentials{}, errors.New("cannot map nil login request")
	}
	return userschema.LoginCredentials{
		Email:    req.Email,
		Password: req.Password,
	}, nil
}

// UserSchemaLoginResultToProto converts userschema.LoginResult to proto.LoginResponse.
func (m *mapperImpl) UserSchemaLoginResultToProto(result *userschema.LoginResult) (*pb.LoginResponse, error) {
	if result == nil {
		return nil, errors.New("cannot map nil login result")
	}
	userProto, err := m.UserEntityToProto(&result.User)
	if err != nil {
		return nil, fmt.Errorf("failed to map user entity to proto: %w", err)
	}
	return &pb.LoginResponse{
		User:         userProto,
		AccessToken:  result.AccessToken,
		RefreshToken: result.RefreshToken,
		ExpiresAt:    result.ExpiresAt,
	}, nil
}

// UserSchemaRefreshResultToProto converts userschema.RefreshResult to proto.RefreshResponse.
func (m *mapperImpl) UserSchemaRefreshResultToProto(result *userschema.RefreshResult) (*pb.RefreshResponse, error) {
	if result == nil {
		return nil, errors.New("cannot map nil refresh result")
	}
	return &pb.RefreshResponse{
		AccessToken:  result.AccessToken,
		RefreshToken: result.RefreshToken,
		ExpiresAt:    result.ExpiresAt,
	}, nil
}

// UserPaginationResultToProtoList converts coreTypes.PaginationResult[entity.User] to proto.ListUsersResponse.
func (m *mapperImpl) UserPaginationResultToProtoList(result *coreTypes.PaginationResult[entity.User]) (*pb.ListUsersResponse, error) {
	if result == nil {
		return &pb.ListUsersResponse{
			Users:          []*pb.User{},
			PaginationInfo: &corePb.PaginationInfo{},
		}, nil
	}

	usersProto := make([]*pb.User, 0, len(result.Items))
	for _, userEntity := range result.Items {
		userProto, err := m.UserEntityToProto(userEntity)
		if err != nil {
			return nil, fmt.Errorf("failed to map user entity %s to proto: %w", userEntity.ID, err)
		}
		usersProto = append(usersProto, userProto)
	}

	paginationInfo := &corePb.PaginationInfo{
		TotalItems: result.TotalItems,
		Limit:      int32(result.Limit),
		Offset:     int32(result.Offset),
	}

	return &pb.ListUsersResponse{
		Users:          usersProto,
		PaginationInfo: paginationInfo,
	}, nil
}

// --- File Mappings (Implementations) ---

func (m *mapperImpl) FileEntityToProto(file *entity.File) (*pb.File, error) {
	if file == nil {
		return nil, errors.New("cannot map nil file entity to proto")
	}
	var deletedAt *timestamppb.Timestamp
	if file.DeletedAt != nil {
		deletedAt = timestamppb.New(*file.DeletedAt)
	}
	return &pb.File{
		Id:                file.ID.String(),
		CreatedAt:         timestamppb.New(file.CreatedAt),
		UpdatedAt:         timestamppb.New(file.UpdatedAt),
		DeletedAt:         deletedAt,
		Name:              file.Name,
		ServiceInternalId: file.ServiceInternalID,
		Type:              file.Type,
		Size:              file.Size,
		Url:               file.URL,
	}, nil
}

func (m *mapperImpl) FilePaginationResultToProtoList(result *coreTypes.PaginationResult[entity.File]) (*pb.ListUserFilesResponse, error) {
	if result == nil {
		return &pb.ListUserFilesResponse{
			Files:          []*pb.File{},
			PaginationInfo: &corePb.PaginationInfo{},
		}, nil
	}

	filesProto := make([]*pb.File, 0, len(result.Items))
	for _, fileEntity := range result.Items {
		fileProto, err := m.FileEntityToProto(fileEntity)
		if err != nil {
			return nil, fmt.Errorf("failed to map file entity %s to proto: %w", fileEntity.ID, err)
		}
		filesProto = append(filesProto, fileProto)
	}

	paginationInfo := &corePb.PaginationInfo{
		TotalItems: result.TotalItems,
		Limit:      int32(result.Limit),
		Offset:     int32(result.Offset),
	}

	return &pb.ListUserFilesResponse{
		Files:          filesProto,
		PaginationInfo: paginationInfo,
	}, nil
}

// FileListToProto maps a slice of File entities to a slice of File protos.
func (m *mapperImpl) FileListToProto(files []*entity.File) ([]*pb.File, error) {
	if files == nil {
		return nil, errors.New("cannot map nil file slice to proto")
	}

	filesProto := make([]*pb.File, 0, len(files))
	for _, fileEntity := range files {
		fileProto, err := m.FileEntityToProto(fileEntity)
		if err != nil {
			return nil, fmt.Errorf("failed to map file entity %s to proto: %w", fileEntity.ID, err)
		}
		filesProto = append(filesProto, fileProto)
	}

	return filesProto, nil
}

// --- Article Mappings (Implementations) ---

func (m *mapperImpl) ArticleEntityToProto(article *entity.Article) (*pb.Article, error) {
	if article == nil {
		return nil, errors.New("cannot map nil article entity to proto")
	}
	var deletedAt *timestamppb.Timestamp
	if article.DeletedAt != nil {
		deletedAt = timestamppb.New(*article.DeletedAt)
	}

	// Populate file_ids by iterating over the Files relationship
	fileIDStrings := make([]string, 0, len(article.Files))
	for _, file := range article.Files {
		fileIDStrings = append(fileIDStrings, file.ID.String()) // Assuming File entity has an ID field
	}

	return &pb.Article{
		Id:         article.ID.String(),
		CreatedAt:  timestamppb.New(article.CreatedAt),
		UpdatedAt:  timestamppb.New(article.UpdatedAt),
		DeletedAt:  deletedAt,
		Title:      article.Title,
		Content:    article.Content,
		AuthorId:   article.AuthorID,
		PictureUrl: article.PictureURL,
		Badge:      article.Badge,
		FileIds:    fileIDStrings, // Populated from article.Files
	}, nil
}

func (m *mapperImpl) ArticleProtoCreateToEntity(req *pb.CreateArticleRequest) (*entity.Article, error) {
	if req == nil {
		return nil, errors.New("cannot map nil create article request to entity")
	}
	// NOTE: We cannot directly map req.FileIds to entity.Files here.
	// The association needs to be handled in the use case/repository layer after the article
	// and the corresponding File entities are created or retrieved.
	// The entity.Files field will typically be populated by GORM's Preload feature when fetching an Article.
	return &entity.Article{
		Title:      req.Title,
		Content:    req.Content,
		AuthorID:   req.AuthorId,
		PictureURL: req.GetPictureUrl(),
		Badge:      req.GetBadge(),
		// Files:  Initialize as empty or nil, relationship handled separately.
	}, nil
}

func (m *mapperImpl) ArticleApplyProtoUpdateToEntity(req *pb.UpdateArticleRequest, existingArticle *entity.Article) error {
	if req == nil || existingArticle == nil {
		return errors.New("update article request and existing entity must not be nil")
	}
	if req.Title != nil {
		existingArticle.Title = req.Title.Value
	}
	if req.Content != nil {
		existingArticle.Content = req.Content.Value
	}
	if req.PictureUrl != nil {
		existingArticle.PictureURL = req.PictureUrl.Value
	}
	if req.Badge != nil {
		existingArticle.Badge = req.Badge.Value
	}
	// NOTE: Updating the Files relationship based on req.FileIds should be handled
	// in the use case/repository layer. Common strategies include:
	// 1. Clearing existing associations and adding new ones based on req.FileIds.Ids.
	// 2. Finding the diff and adding/removing specific associations.
	// The mapper should only update the direct fields of the Article entity.
	// if req.FileIds != nil { ... logic to update existingArticle.Files ... } // <-- This logic belongs elsewhere
	return nil
}

func (m *mapperImpl) ArticlePaginationResultToProtoList(result *coreTypes.PaginationResult[entity.Article]) (*pb.ListArticlesResponse, error) {
	if result == nil {
		return &pb.ListArticlesResponse{
			Articles:       []*pb.Article{},
			PaginationInfo: &corePb.PaginationInfo{},
		}, nil
	}

	articlesProto := make([]*pb.Article, 0, len(result.Items))
	for _, articleEntity := range result.Items {
		articleProto, err := m.ArticleEntityToProto(articleEntity)
		if err != nil {
			return nil, fmt.Errorf("failed to map article entity %s to proto: %w", articleEntity.ID, err)
		}
		articlesProto = append(articlesProto, articleProto)
	}

	paginationInfo := &corePb.PaginationInfo{
		TotalItems: result.TotalItems,
		Limit:      int32(result.Limit),
		Offset:     int32(result.Offset),
	}

	return &pb.ListArticlesResponse{
		Articles:       articlesProto,
		PaginationInfo: paginationInfo,
	}, nil
}

// --- Request Mappings (Implementations) ---

func (m *mapperImpl) RequestEntityToProto(request *entity.Request) (*pb.Request, error) {
	if request == nil {
		return nil, errors.New("cannot map nil request entity to proto")
	}
	var deletedAt *timestamppb.Timestamp
	if request.DeletedAt != nil {
		deletedAt = timestamppb.New(*request.DeletedAt)
	}

	// Populate file_ids by iterating over the Files relationship
	fileIDStrings := make([]string, 0, len(request.Files))
	for _, file := range request.Files {
		fileIDStrings = append(fileIDStrings, file.ID.String()) // Assuming File entity has an ID field
	}

	return &pb.Request{
		Id:          request.ID.String(),
		CreatedAt:   timestamppb.New(request.CreatedAt),
		UpdatedAt:   timestamppb.New(request.UpdatedAt),
		DeletedAt:   deletedAt,
		Title:       request.Title,
		Description: request.Description,
		SenderId:    request.SenderID.String(),
		ReceiverId:  request.ReceiverID.String(),
		Status:      request.Status,
		FileIds:     fileIDStrings, // Populated from request.Files
	}, nil
}

func (m *mapperImpl) RequestProtoCreateToEntity(req *pb.CreateRequestRequest) (*entity.Request, error) {
	if req == nil {
		return nil, errors.New("cannot map nil create request request to entity")
	}
	senderID, err := uuid.Parse(req.SenderId)
	if err != nil {
		return nil, fmt.Errorf("invalid sender_id format: %w", err)
	}
	receiverID, err := uuid.Parse(req.ReceiverId)
	if err != nil {
		return nil, fmt.Errorf("invalid receiver_id format: %w", err)
	}
	// NOTE: We cannot directly map req.FileIds to entity.Files here.
	// The association needs to be handled in the use case/repository layer after the request
	// and the corresponding File entities are created or retrieved.
	// The entity.Files field will typically be populated by GORM's Preload feature when fetching a Request.
	return &entity.Request{
		Title:       req.Title,
		Description: req.GetDescription(),
		SenderID:    senderID,
		ReceiverID:  receiverID,
		Status:      req.Status,
		// Files: Initialize as empty or nil, relationship handled separately.
	}, nil
}

func (m *mapperImpl) RequestApplyProtoUpdateToEntity(req *pb.UpdateRequestRequest, existingRequest *entity.Request) error {
	if req == nil || existingRequest == nil {
		return errors.New("update request request and existing entity must not be nil")
	}
	if req.Title != nil {
		existingRequest.Title = req.Title.Value
	}
	if req.Status != nil {
		existingRequest.Status = req.Status.Value
	}
	if req.Description != nil {
		existingRequest.Description = req.Description.Value
	}
	// NOTE: Updating the Files relationship based on req.FileIds should be handled
	// in the use case/repository layer. Common strategies include:
	// 1. Clearing existing associations and adding new ones based on req.FileIds.Ids.
	// 2. Finding the diff and adding/removing specific associations.
	// The mapper should only update the direct fields of the Request entity.
	// if req.FileIds != nil { ... logic to update existingRequest.Files ... } // <-- This logic belongs elsewhere
	return nil
}

func (m *mapperImpl) RequestPaginationResultToProtoList(result *coreTypes.PaginationResult[entity.Request]) (*pb.ListRequestsResponse, error) {
	if result == nil {
		return &pb.ListRequestsResponse{
			Requests:       []*pb.Request{},
			PaginationInfo: &corePb.PaginationInfo{},
		}, nil
	}

	requestsProto := make([]*pb.Request, 0, len(result.Items))
	for _, requestEntity := range result.Items {
		requestProto, err := m.RequestEntityToProto(requestEntity)
		if err != nil {
			return nil, fmt.Errorf("failed to map request entity %s to proto: %w", requestEntity.ID, err)
		}
		requestsProto = append(requestsProto, requestProto)
	}

	paginationInfo := &corePb.PaginationInfo{
		TotalItems: result.TotalItems,
		Limit:      int32(result.Limit),
		Offset:     int32(result.Offset),
	}

	return &pb.ListRequestsResponse{
		Requests:       requestsProto,
		PaginationInfo: paginationInfo,
	}, nil
}

// --- Chat Mappings (Implementations) ---

func (m *mapperImpl) ChatMessageEntityToProto(msg *entity.ChatMessage) (*pb.DirectChatMessage, error) { // Corrected return type
	if msg == nil {
		return nil, errors.New("cannot map nil chat message entity to proto")
	}
	// ConversationId is not part of DirectChatMessage proto
	// Need UpdatedAt and DeletedAt from BaseEntity if included in proto
	var updatedAt *timestamppb.Timestamp
	if !msg.UpdatedAt.IsZero() { // Check if UpdatedAt is set
		updatedAt = timestamppb.New(msg.UpdatedAt)
	}
	var deletedAt *timestamppb.Timestamp
	if msg.DeletedAt != nil {
		deletedAt = timestamppb.New(*msg.DeletedAt)
	}

	return &pb.DirectChatMessage{ // Corrected type
		Id:         msg.ID.String(),
		CreatedAt:  timestamppb.New(msg.CreatedAt),
		SenderId:   msg.SenderID.String(),   // Assuming SenderID is uuid.UUID in entity
		ReceiverId: msg.ReceiverID.String(), // Assuming ReceiverID is uuid.UUID in entity
		Message:    msg.Message,
		Read:       msg.Read,
		UpdatedAt:  updatedAt, // Map updated_at
		DeletedAt:  deletedAt, // Map deleted_at
	}, nil
}

// --- Notification Mappings (Implementations) ---

func (m *mapperImpl) NotificationEntityToProto(notification *entity.Notification) (*pb.Notification, error) {
	if notification == nil {
		return nil, errors.New("cannot map nil notification entity to proto")
	}
	var deletedAt *timestamppb.Timestamp
	if notification.DeletedAt != nil {
		deletedAt = timestamppb.New(*notification.DeletedAt)
	}
	return &pb.Notification{
		Id:          notification.ID.String(),
		CreatedAt:   timestamppb.New(notification.CreatedAt),
		UpdatedAt:   timestamppb.New(notification.UpdatedAt),
		DeletedAt:   deletedAt,
		Title:       notification.Title,
		Description: notification.Description,
		Read:        notification.Read,
		UserId:      notification.UserID.String(),
	}, nil
}

func (m *mapperImpl) NotificationProtoCreateToEntity(req *pb.CreateNotificationRequest) (*entity.Notification, error) {
	if req == nil {
		return nil, errors.New("cannot map nil create notification request to entity")
	}
	userID, err := uuid.Parse(req.UserId)
	if err != nil {
		return nil, fmt.Errorf("invalid user_id format: %w", err)
	}
	return &entity.Notification{
		Title:       req.Title,
		Description: req.Description,
		UserID:      userID,
		Read:        false,
	}, nil
}

func (m *mapperImpl) NotificationApplyProtoUpdateToEntity(req *pb.UpdateNotificationRequest, existingNotification *entity.Notification) error {
	if req == nil || existingNotification == nil {
		return errors.New("update notification request and existing entity must not be nil")
	}
	if req.Title != nil {
		existingNotification.Title = req.Title.Value
	}
	if req.Description != nil {
		existingNotification.Description = req.Description.Value
	}
	if req.Read != nil {
		existingNotification.Read = req.Read.Value
	}
	return nil
}

func (m *mapperImpl) NotificationPaginationResultToProtoList(result *coreTypes.PaginationResult[entity.Notification]) (*pb.ListNotificationsResponse, error) {
	if result == nil {
		return &pb.ListNotificationsResponse{
			Notifications:  []*pb.Notification{},
			PaginationInfo: &corePb.PaginationInfo{},
		}, nil
	}

	notificationsProto := make([]*pb.Notification, 0, len(result.Items))
	for _, notificationEntity := range result.Items {
		notificationProto, err := m.NotificationEntityToProto(notificationEntity)
		if err != nil {
			return nil, fmt.Errorf("failed to map notification entity %s to proto: %w", notificationEntity.ID, err)
		}
		notificationsProto = append(notificationsProto, notificationProto)
	}

	paginationInfo := &corePb.PaginationInfo{
		TotalItems: result.TotalItems,
		Limit:      int32(result.Limit),
		Offset:     int32(result.Offset),
	}

	return &pb.ListNotificationsResponse{
		Notifications:  notificationsProto,
		PaginationInfo: paginationInfo,
	}, nil
}

// --- Common Mappings ---

func (m *mapperImpl) ProtoListRequestToFilterOptions(options *corePb.FilterOptions) coreTypes.FilterOptions {
	opts := coreTypes.DefaultFilterOptions()
	if options == nil {
		return opts
	}

	if options.Limit != nil {
		opts.Limit = int(*options.Limit)
	}
	if options.Offset != nil {
		opts.Offset = int(*options.Offset)
	}
	if options.SortBy != nil {
		opts.SortBy = *options.SortBy
	}
	if options.SortDesc != nil {
		opts.SortDesc = *options.SortDesc
	}
	if options.IncludeDeleted != nil {
		opts.IncludeDeleted = *options.IncludeDeleted
	}

	if len(options.Filters) > 0 {
		opts.Filters = make(map[string]interface{}, len(options.Filters))
		for k, v := range options.Filters {
			opts.Filters[k] = mapProtoValueToGo(v)
		}
	}
	return opts
}

func mapProtoValueToGo(v *structpb.Value) interface{} {
	if v == nil {
		return nil
	}
	switch kind := v.Kind.(type) {
	case *structpb.Value_NullValue:
		return nil
	case *structpb.Value_NumberValue:
		return kind.NumberValue
	case *structpb.Value_StringValue:
		return kind.StringValue
	case *structpb.Value_BoolValue:
		return kind.BoolValue
	case *structpb.Value_StructValue:
		fields := make(map[string]interface{})
		for key, val := range kind.StructValue.Fields {
			fields[key] = mapProtoValueToGo(val)
		}
		return fields
	case *structpb.Value_ListValue:
		list := make([]interface{}, 0, len(kind.ListValue.Values))
		for _, val := range kind.ListValue.Values {
			list = append(list, mapProtoValueToGo(val))
		}
		return list
	default:
		return nil
	}
}

// Helper function to safely dereference string pointers
func derefString(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}
