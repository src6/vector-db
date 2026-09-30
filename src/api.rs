use std::collections::HashMap;

use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::{ConcurrentIndex, InMemoryStorage, IndexError, Metric, Neighbor, PointId};

const MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ApiConfig {
    pub dimension: usize,
    pub metric: Metric,
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

#[derive(Clone)]
struct ApiState {
    index: ConcurrentIndex<InMemoryStorage>,
    config: ApiConfig,
}

#[derive(Debug, Deserialize)]
struct InsertRequest {
    vector: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct InsertBatchRequest {
    vectors: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct InsertResponse {
    id: PointId,
}

#[derive(Debug, Serialize)]
struct InsertBatchResponse {
    ids: Vec<PointId>,
}

#[derive(Debug, Deserialize)]
struct SearchRequest {
    vector: Vec<f32>,
    k: usize,
}

#[derive(Debug, Deserialize)]
struct SearchBatchRequest {
    queries: Vec<Vec<f32>>,
    k: usize,
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    neighbors: Vec<Neighbor>,
}

#[derive(Debug, Serialize)]
struct SearchBatchResponse {
    results: Vec<Vec<Neighbor>>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
}

#[derive(Debug, Serialize)]
struct StatsResponse {
    total_vectors: usize,
    active_vectors: usize,
    deleted_vectors: usize,
    #[serde(flatten)]
    config: ApiConfig,
}

#[derive(Debug, Serialize)]
struct IdMapping {
    old_id: PointId,
    new_id: PointId,
}

#[derive(Debug, Serialize)]
struct CompactResponse {
    removed: usize,
    id_map: Vec<IdMapping>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: &'static str,
    message: String,
}

enum ApiError {
    Index(IndexError),
    Internal,
}

impl From<IndexError> for ApiError {
    fn from(error: IndexError) -> Self {
        Self::Index(error)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            Self::Index(error @ IndexError::PointNotFound { .. }) => {
                (StatusCode::NOT_FOUND, "point_not_found", error.to_string())
            }
            Self::Index(error @ IndexError::PointAlreadyDeleted { .. }) => (
                StatusCode::CONFLICT,
                "point_already_deleted",
                error.to_string(),
            ),
            Self::Index(error) => (
                StatusCode::BAD_REQUEST,
                "invalid_request",
                error.to_string(),
            ),
            Self::Internal => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "blocking index task failed".to_owned(),
            ),
        };
        (
            status,
            Json(ErrorResponse {
                error: code,
                message,
            }),
        )
            .into_response()
    }
}

pub fn router(index: ConcurrentIndex<InMemoryStorage>, config: ApiConfig) -> Router {
    let state = ApiState { index, config };
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/stats", get(stats))
        .route("/v1/vectors", post(insert))
        .route("/v1/vectors/batch", post(insert_batch))
        .route("/v1/vectors/{id}", delete(delete_vector))
        .route("/v1/search", post(search))
        .route("/v1/search/batch", post(search_batch))
        .route("/v1/maintenance/compact", post(compact))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
        .with_state(state)
}

fn validate_request_vector(config: ApiConfig, vector: &[f32]) -> Result<(), ApiError> {
    if vector.len() != config.dimension {
        return Err(ApiError::Index(IndexError::DimensionMismatch {
            expected: config.dimension,
            actual: vector.len(),
        }));
    }
    Ok(())
}

fn validate_k(k: usize) -> Result<(), ApiError> {
    if k == 0 {
        return Err(ApiError::Index(IndexError::InvalidConfiguration(
            "k must be greater than zero",
        )));
    }
    Ok(())
}

async fn run_index_task<T, F>(task: F) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, IndexError> + Send + 'static,
{
    tokio::task::spawn_blocking(task)
        .await
        .map_err(|_| ApiError::Internal)?
        .map_err(ApiError::from)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn stats(State(state): State<ApiState>) -> Json<StatsResponse> {
    let (total_vectors, active_vectors, deleted_vectors) = state.index.counts();
    Json(StatsResponse {
        total_vectors,
        active_vectors,
        deleted_vectors,
        config: state.config,
    })
}

async fn insert(
    State(state): State<ApiState>,
    Json(request): Json<InsertRequest>,
) -> Result<(StatusCode, Json<InsertResponse>), ApiError> {
    validate_request_vector(state.config, &request.vector)?;
    let id = run_index_task(move || state.index.try_insert(request.vector)).await?;
    Ok((StatusCode::CREATED, Json(InsertResponse { id })))
}

async fn insert_batch(
    State(state): State<ApiState>,
    Json(request): Json<InsertBatchRequest>,
) -> Result<(StatusCode, Json<InsertBatchResponse>), ApiError> {
    for vector in &request.vectors {
        validate_request_vector(state.config, vector)?;
    }
    let ids = run_index_task(move || state.index.try_insert_batch(request.vectors)).await?;
    Ok((StatusCode::CREATED, Json(InsertBatchResponse { ids })))
}

async fn search(
    State(state): State<ApiState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    validate_request_vector(state.config, &request.vector)?;
    validate_k(request.k)?;
    let neighbors =
        run_index_task(move || state.index.try_search(&request.vector, request.k)).await?;
    Ok(Json(SearchResponse { neighbors }))
}

async fn search_batch(
    State(state): State<ApiState>,
    Json(request): Json<SearchBatchRequest>,
) -> Result<Json<SearchBatchResponse>, ApiError> {
    validate_k(request.k)?;
    for query in &request.queries {
        validate_request_vector(state.config, query)?;
    }
    let results = run_index_task(move || {
        state
            .index
            .try_search_batch_parallel(request.queries, request.k)
    })
    .await?;
    Ok(Json(SearchBatchResponse { results }))
}

async fn delete_vector(
    State(state): State<ApiState>,
    Path(id): Path<PointId>,
) -> Result<StatusCode, ApiError> {
    run_index_task(move || state.index.delete(id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn compact(State(state): State<ApiState>) -> Result<Json<CompactResponse>, ApiError> {
    let (removed, mapping): (usize, HashMap<_, _>) =
        run_index_task(move || state.index.compact()).await?;
    let mut id_map: Vec<_> = mapping
        .into_iter()
        .map(|(old_id, new_id)| IdMapping { old_id, new_id })
        .collect();
    id_map.sort_by_key(|mapping| mapping.old_id);
    Ok(Json(CompactResponse { removed, id_map }))
}

#[cfg(test)]
mod tests {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use super::*;
    use crate::HnswIndex;

    fn test_app() -> Router {
        let config = ApiConfig {
            dimension: 2,
            metric: Metric::L2,
            m: 8,
            m_max0: 16,
            ef_construction: 32,
            ef_search: 32,
        };
        let index = HnswIndex::new(
            config.m,
            config.m_max0,
            config.ef_construction,
            config.ef_search,
            config.metric,
            InMemoryStorage::new(),
        )
        .with_level_seed(42);
        router(ConcurrentIndex::new(index), config)
    }

    async fn request(app: &Router, method: &str, uri: &str, body: Value) -> (StatusCode, Value) {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap()
        };
        (status, body)
    }

    #[tokio::test]
    async fn vector_lifecycle_includes_logical_delete_and_compaction() {
        let app = test_app();
        let (status, _) = request(
            &app,
            "POST",
            "/v1/vectors/batch",
            json!({ "vectors": [[0.0, 0.0], [1.0, 0.0]] }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);

        let (status, _) = request(&app, "DELETE", "/v1/vectors/0", Value::Null).await;
        assert_eq!(status, StatusCode::NO_CONTENT);

        let (_, search_body) = request(
            &app,
            "POST",
            "/v1/search",
            json!({ "vector": [0.0, 0.0], "k": 2 }),
        )
        .await;
        assert_eq!(search_body["neighbors"][0]["id"], 1);
        assert_eq!(search_body["neighbors"].as_array().unwrap().len(), 1);

        let (_, stats_body) = request(&app, "GET", "/v1/stats", Value::Null).await;
        assert_eq!(stats_body["total_vectors"], 2);
        assert_eq!(stats_body["active_vectors"], 1);
        assert_eq!(stats_body["deleted_vectors"], 1);

        let (_, compact_body) = request(&app, "POST", "/v1/maintenance/compact", Value::Null).await;
        assert_eq!(compact_body["removed"], 1);
        assert_eq!(
            compact_body["id_map"],
            json!([{ "old_id": 1, "new_id": 0 }])
        );
    }

    #[tokio::test]
    async fn invalid_dimensions_return_structured_errors() {
        let app = test_app();
        let (status, body) = request(&app, "POST", "/v1/vectors", json!({ "vector": [1.0] })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"], "invalid_request");
    }
}
