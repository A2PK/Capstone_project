package controller

import (
	"context"

	"github.com/google/uuid"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/emptypb"

	coreController "golang-microservices-boilerplate/pkg/core/controller"
	// coreTypes "golang-microservices-boilerplate/pkg/core/types" // Import if needed
	pb "golang-microservices-boilerplate/proto/user-service"
	"golang-microservices-boilerplate/services/user-service/internal/usecase"
)

// Ensure articleServer implements pb.ArticleServiceServer.
var _ pb.ArticleServiceServer = (*articleServer)(nil)

type articleServer struct {
	pb.UnimplementedArticleServiceServer
	uc     usecase.ArticleUsecase
	mapper Mapper // Use the general Mapper
}

// NewArticleServer creates a new gRPC article server instance.
func NewArticleServer(uc usecase.ArticleUsecase, mapper Mapper) pb.ArticleServiceServer {
	return &articleServer{
		uc:     uc,
		mapper: mapper,
	}
}

// RegisterArticleServiceServer registers the article service implementation.
func RegisterArticleServiceServer(s *grpc.Server, uc usecase.ArticleUsecase, mapper Mapper) {
	server := NewArticleServer(uc, mapper)
	pb.RegisterArticleServiceServer(s, server)
}

// --- Implement ArticleServiceServer Methods ---

func (s *articleServer) Create(ctx context.Context, req *pb.CreateArticleRequest) (*pb.CreateArticleResponse, error) {
	// Map proto request to entity
	articleEntity, err := s.mapper.ArticleProtoCreateToEntity(req)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "failed to map request: %v", err)
	}

	articleEntity, err = s.uc.CreateArticleWithFiles(ctx, articleEntity, req.GetFileIds())
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}
	// Map result entity back to proto
	articleProto, err := s.mapper.ArticleEntityToProto(articleEntity)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to map result: %v", err)
	}

	return &pb.CreateArticleResponse{Article: articleProto}, nil
}

func (s *articleServer) GetByID(ctx context.Context, req *pb.GetArticleByIDRequest) (*pb.GetArticleByIDResponse, error) {
	id, err := uuid.Parse(req.GetId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid article ID format: %v", err)
	}

	articleEntity, err := s.uc.GetByID(ctx, id) // Assuming base GetByID method
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	articleProto, err := s.mapper.ArticleEntityToProto(articleEntity)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to map result: %v", err)
	}

	return &pb.GetArticleByIDResponse{Article: articleProto}, nil
}

func (s *articleServer) List(ctx context.Context, req *pb.ListArticlesRequest) (*pb.ListArticlesResponse, error) {
	opts := s.mapper.ProtoListRequestToFilterOptions(req.Options)

	result, err := s.uc.List(ctx, opts) // Assuming base List handles filters
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	response, err := s.mapper.ArticlePaginationResultToProtoList(result)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to map result list: %v", err)
	}

	return response, nil
}

func (s *articleServer) Update(ctx context.Context, req *pb.UpdateArticleRequest) (*pb.UpdateArticleResponse, error) {
	id, err := uuid.Parse(req.GetId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid article ID format: %v", err)
	}

	// 1. Get existing entity
	existingArticle, err := s.uc.GetByID(ctx, id) // Assuming base GetByID
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// TODO: Authorization Check: Can the current user update this article? (e.g., check AuthorID)

	// 2. Apply updates from proto request
	if err := s.mapper.ArticleApplyProtoUpdateToEntity(req, existingArticle); err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "failed to map update request: %v", err)
	}

	// 3. Call use case Update
	err = s.uc.Update(ctx, existingArticle) // Assuming base Update
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	// 4. Map updated entity back to proto
	articleProto, err := s.mapper.ArticleEntityToProto(existingArticle)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to map result: %v", err)
	}

	return &pb.UpdateArticleResponse{Article: articleProto}, nil
}

func (s *articleServer) Delete(ctx context.Context, req *pb.DeleteArticleRequest) (*emptypb.Empty, error) {
	id, err := uuid.Parse(req.GetId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid article ID format: %v", err)
	}
	hardDelete := req.GetHardDelete()

	// TODO: Authorization Check: Can the current user delete this article?

	err = s.uc.Delete(ctx, id, hardDelete) // Assuming base Delete
	if err != nil {
		return nil, coreController.MapErrorToHttpStatus(err)
	}

	return &emptypb.Empty{}, nil
}
