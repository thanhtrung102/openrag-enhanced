-- OpenRAG Enhanced - Evaluation Database Schema
-- PostgreSQL initialization script

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Evaluation runs table
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id SERIAL PRIMARY KEY,
    run_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
    run_type VARCHAR(50) NOT NULL,  -- 'retrieval', 'generation', 'hallucination', 'benchmark', 'full'
    model_name VARCHAR(100),
    retrieval_method VARCHAR(50),
    config JSONB,  -- Store evaluation configuration
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'running',  -- 'running', 'completed', 'failed'
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Query logs for tracking all interactions
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    query_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
    run_id UUID REFERENCES evaluation_runs(run_id) ON DELETE SET NULL,
    query_text TEXT NOT NULL,
    query_embedding BYTEA,  -- Store embedding for similarity analysis
    retrieval_method VARCHAR(50),
    model_name VARCHAR(100),
    session_id UUID,
    user_id VARCHAR(100),
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Retrieval evaluation results
CREATE TABLE IF NOT EXISTS retrieval_evaluations (
    id SERIAL PRIMARY KEY,
    run_id UUID REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
    query_id UUID REFERENCES query_logs(query_id) ON DELETE CASCADE,
    query_text TEXT,
    -- Hit rate metrics
    hit_rate_1 FLOAT,
    hit_rate_3 FLOAT,
    hit_rate_5 FLOAT,
    hit_rate_10 FLOAT,
    hit_rate_20 FLOAT,
    -- Ranking metrics
    mrr FLOAT,  -- Mean Reciprocal Rank
    ndcg_5 FLOAT,
    ndcg_10 FLOAT,
    ndcg_20 FLOAT,
    -- Precision/Recall
    precision_5 FLOAT,
    precision_10 FLOAT,
    recall_5 FLOAT,
    recall_10 FLOAT,
    -- Performance metrics
    latency_ms INTEGER,
    num_results INTEGER,
    -- Document tracking
    retrieved_doc_ids TEXT[],
    relevant_doc_ids TEXT[],
    retrieval_scores FLOAT[],
    -- Metadata
    retrieval_method VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Generation/RAG evaluation results (LLM-as-Judge scores)
CREATE TABLE IF NOT EXISTS generation_evaluations (
    id SERIAL PRIMARY KEY,
    run_id UUID REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
    query_id UUID REFERENCES query_logs(query_id) ON DELETE CASCADE,
    query_text TEXT,
    response_text TEXT,
    context_texts TEXT[],  -- Retrieved context used for generation
    reference_answer TEXT,  -- Ground truth if available
    -- LLM-as-Judge scores (0-5 scale)
    faithfulness_score FLOAT,
    answer_relevance_score FLOAT,
    context_relevance_score FLOAT,
    completeness_score FLOAT,
    coherence_score FLOAT,
    overall_score FLOAT,
    -- Judge metadata
    judge_model VARCHAR(50),
    judge_explanation TEXT,
    judge_raw_response JSONB,
    -- Performance
    generation_latency_ms INTEGER,
    tokens_used INTEGER,
    -- Metadata
    model_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Hallucination detection results
CREATE TABLE IF NOT EXISTS hallucination_detections (
    id SERIAL PRIMARY KEY,
    run_id UUID REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
    query_id UUID REFERENCES query_logs(query_id) ON DELETE CASCADE,
    response_text TEXT,
    source_texts TEXT[],  -- Context used for detection
    -- Detection scores by method
    nli_score FLOAT,  -- Natural Language Inference score
    nli_details JSONB,  -- Per-claim NLI results
    alignment_score FLOAT,  -- Source-answer alignment
    alignment_details JSONB,
    consistency_score FLOAT,  -- Self-consistency score
    consistency_details JSONB,
    -- Ensemble result
    ensemble_score FLOAT,
    ensemble_method VARCHAR(50),  -- 'average', 'weighted', 'voting'
    is_hallucinated BOOLEAN,
    confidence FLOAT,
    -- Flagged content
    flagged_claims TEXT[],
    claim_scores JSONB,  -- Detailed per-claim scores
    -- Human verification
    human_verified BOOLEAN DEFAULT FALSE,
    human_label VARCHAR(20),  -- 'hallucinated', 'accurate', 'partial'
    human_notes TEXT,
    verified_by VARCHAR(100),
    verified_at TIMESTAMP,
    -- Metadata
    detection_method VARCHAR(50),
    model_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- User feedback collection
CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    feedback_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
    query_id UUID REFERENCES query_logs(query_id) ON DELETE CASCADE,
    -- Feedback data
    feedback_type VARCHAR(20) NOT NULL,  -- 'accurate', 'hallucinated', 'partial', 'helpful', 'not_helpful'
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    comment TEXT,
    -- Specific feedback
    flagged_content TEXT,  -- User-highlighted problematic content
    suggested_correction TEXT,
    -- User info
    user_id VARCHAR(100),
    session_id UUID,
    -- Metadata
    source VARCHAR(50),  -- 'chainlit', 'api', 'dashboard'
    created_at TIMESTAMP DEFAULT NOW()
);

-- Ground truth dataset storage
CREATE TABLE IF NOT EXISTS ground_truth_datasets (
    id SERIAL PRIMARY KEY,
    dataset_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    version VARCHAR(50),
    source VARCHAR(200),  -- Where the dataset came from
    language VARCHAR(10) DEFAULT 'en',
    domain VARCHAR(100),  -- 'technical', 'general', 'medical', etc.
    total_pairs INTEGER,
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Ground truth Q&A pairs
CREATE TABLE IF NOT EXISTS ground_truth_pairs (
    id SERIAL PRIMARY KEY,
    pair_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
    dataset_id UUID REFERENCES ground_truth_datasets(dataset_id) ON DELETE CASCADE,
    -- Question and answer
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    -- Document reference
    document_id VARCHAR(200),
    document_title VARCHAR(500),
    supporting_quote TEXT,  -- Exact quote from document
    -- Metadata
    difficulty VARCHAR(20),  -- 'easy', 'medium', 'hard'
    question_type VARCHAR(50),  -- 'factual', 'reasoning', 'comparison', 'multi-hop'
    requires_multi_hop BOOLEAN DEFAULT FALSE,
    -- Validation
    manually_verified BOOLEAN DEFAULT FALSE,
    verified_by VARCHAR(100),
    quality_score FLOAT,
    -- Additional info
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Benchmark definitions
CREATE TABLE IF NOT EXISTS benchmarks (
    id SERIAL PRIMARY KEY,
    benchmark_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    version VARCHAR(50),
    -- Configuration
    config JSONB,  -- Benchmark configuration
    metrics TEXT[],  -- List of metrics to compute
    -- Dataset reference
    dataset_id UUID REFERENCES ground_truth_datasets(dataset_id),
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Benchmark run results
CREATE TABLE IF NOT EXISTS benchmark_results (
    id SERIAL PRIMARY KEY,
    result_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
    benchmark_id UUID REFERENCES benchmarks(benchmark_id) ON DELETE CASCADE,
    run_id UUID REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
    -- Results
    metrics JSONB NOT NULL,  -- All computed metrics
    summary TEXT,
    -- Comparison
    baseline_result_id UUID REFERENCES benchmark_results(result_id),
    improvement_vs_baseline JSONB,
    -- Metadata
    model_name VARCHAR(100),
    retrieval_method VARCHAR(50),
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- Materialized Views for Dashboard Performance
-- ============================================

-- Daily evaluation summary
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_evaluation_summary AS
SELECT
    DATE(re.created_at) as date,
    er.model_name,
    er.retrieval_method,
    COUNT(DISTINCT re.id) as retrieval_evals,
    COUNT(DISTINCT ge.id) as generation_evals,
    COUNT(DISTINCT hd.id) as hallucination_checks,
    -- Retrieval metrics
    AVG(re.hit_rate_5) as avg_hit_rate_5,
    AVG(re.mrr) as avg_mrr,
    AVG(re.ndcg_10) as avg_ndcg_10,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY re.latency_ms) as median_retrieval_latency,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY re.latency_ms) as p95_retrieval_latency,
    -- Generation metrics
    AVG(ge.faithfulness_score) as avg_faithfulness,
    AVG(ge.answer_relevance_score) as avg_relevance,
    AVG(ge.overall_score) as avg_overall_score,
    -- Hallucination metrics
    AVG(hd.ensemble_score) as avg_hallucination_score,
    SUM(CASE WHEN hd.is_hallucinated THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(COUNT(hd.id), 0) as hallucination_rate
FROM evaluation_runs er
LEFT JOIN retrieval_evaluations re ON er.run_id = re.run_id
LEFT JOIN generation_evaluations ge ON er.run_id = ge.run_id
LEFT JOIN hallucination_detections hd ON er.run_id = hd.run_id
WHERE er.created_at > NOW() - INTERVAL '90 days'
GROUP BY DATE(re.created_at), er.model_name, er.retrieval_method;

-- Model performance comparison
CREATE MATERIALIZED VIEW IF NOT EXISTS model_performance_summary AS
SELECT
    ge.model_name,
    COUNT(*) as total_evaluations,
    AVG(ge.faithfulness_score) as avg_faithfulness,
    AVG(ge.answer_relevance_score) as avg_relevance,
    AVG(ge.completeness_score) as avg_completeness,
    AVG(ge.coherence_score) as avg_coherence,
    AVG(ge.overall_score) as avg_overall,
    STDDEV(ge.overall_score) as stddev_overall,
    AVG(ge.generation_latency_ms) as avg_latency_ms,
    AVG(ge.tokens_used) as avg_tokens,
    -- Hallucination stats
    AVG(hd.ensemble_score) as avg_hallucination_score,
    SUM(CASE WHEN hd.is_hallucinated THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(COUNT(hd.id), 0) as hallucination_rate,
    -- User feedback
    COUNT(uf.id) as feedback_count,
    AVG(uf.rating) as avg_user_rating,
    SUM(CASE WHEN uf.feedback_type = 'accurate' THEN 1 ELSE 0 END)::FLOAT /
        NULLIF(COUNT(uf.id), 0) as accuracy_rate_user
FROM generation_evaluations ge
LEFT JOIN hallucination_detections hd ON ge.query_id = hd.query_id
LEFT JOIN user_feedback uf ON ge.query_id = uf.query_id
WHERE ge.created_at > NOW() - INTERVAL '30 days'
GROUP BY ge.model_name;

-- ============================================
-- Indexes for Query Performance
-- ============================================

CREATE INDEX IF NOT EXISTS idx_eval_runs_created ON evaluation_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_eval_runs_status ON evaluation_runs(status);
CREATE INDEX IF NOT EXISTS idx_eval_runs_type ON evaluation_runs(run_type);

CREATE INDEX IF NOT EXISTS idx_query_logs_timestamp ON query_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_query_logs_session ON query_logs(session_id);

CREATE INDEX IF NOT EXISTS idx_retrieval_run_id ON retrieval_evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_created ON retrieval_evaluations(created_at);
CREATE INDEX IF NOT EXISTS idx_retrieval_method ON retrieval_evaluations(retrieval_method);

CREATE INDEX IF NOT EXISTS idx_generation_run_id ON generation_evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_generation_created ON generation_evaluations(created_at);
CREATE INDEX IF NOT EXISTS idx_generation_model ON generation_evaluations(model_name);

CREATE INDEX IF NOT EXISTS idx_hallucination_run_id ON hallucination_detections(run_id);
CREATE INDEX IF NOT EXISTS idx_hallucination_created ON hallucination_detections(created_at);
CREATE INDEX IF NOT EXISTS idx_hallucination_is_hall ON hallucination_detections(is_hallucinated);

CREATE INDEX IF NOT EXISTS idx_feedback_query ON user_feedback(query_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON user_feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON user_feedback(feedback_type);

CREATE INDEX IF NOT EXISTS idx_gt_pairs_dataset ON ground_truth_pairs(dataset_id);
CREATE INDEX IF NOT EXISTS idx_gt_pairs_type ON ground_truth_pairs(question_type);

-- Full text search on queries
CREATE INDEX IF NOT EXISTS idx_query_text_search ON query_logs USING gin(to_tsvector('english', query_text));

-- ============================================
-- Functions for Evaluation
-- ============================================

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_evaluation_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_evaluation_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY model_performance_summary;
END;
$$ LANGUAGE plpgsql;

-- Function to get model comparison
CREATE OR REPLACE FUNCTION get_model_comparison(
    start_date TIMESTAMP DEFAULT NOW() - INTERVAL '7 days',
    end_date TIMESTAMP DEFAULT NOW()
)
RETURNS TABLE (
    model_name VARCHAR,
    avg_faithfulness FLOAT,
    avg_relevance FLOAT,
    hallucination_rate FLOAT,
    avg_latency_ms FLOAT,
    total_queries BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ge.model_name,
        AVG(ge.faithfulness_score)::FLOAT,
        AVG(ge.answer_relevance_score)::FLOAT,
        (SUM(CASE WHEN hd.is_hallucinated THEN 1 ELSE 0 END)::FLOAT /
            NULLIF(COUNT(hd.id), 0))::FLOAT,
        AVG(ge.generation_latency_ms)::FLOAT,
        COUNT(DISTINCT ge.query_id)
    FROM generation_evaluations ge
    LEFT JOIN hallucination_detections hd ON ge.query_id = hd.query_id
    WHERE ge.created_at BETWEEN start_date AND end_date
    GROUP BY ge.model_name
    ORDER BY AVG(ge.overall_score) DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- Initial Data (Optional)
-- ============================================

-- Insert default benchmark configuration
INSERT INTO benchmarks (name, description, version, config, metrics, is_active)
VALUES (
    'OpenRAG Standard Benchmark',
    'Standard evaluation benchmark for OpenRAG systems',
    '1.0.0',
    '{
        "retrieval": {"k_values": [1, 3, 5, 10, 20]},
        "generation": {"judge_model": "gpt-4", "temperature": 0},
        "hallucination": {"methods": ["nli", "alignment", "consistency"]}
    }'::jsonb,
    ARRAY['hit_rate', 'mrr', 'ndcg', 'faithfulness', 'hallucination_rate'],
    true
) ON CONFLICT DO NOTHING;

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO openrag_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO openrag_app;
