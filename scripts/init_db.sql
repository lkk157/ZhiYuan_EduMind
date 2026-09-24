-- ============================================================
-- 知源 · MySQL 初始化脚本（幂等：可反复执行不报错）
-- 用法：mysql -u root -p < scripts/init_db.sql
--
-- 为什么统一 utf8mb4：MySQL 8 默认 utf8mb3 存不下 emoji/生僻字且中文易乱码，
-- 建库建表连接三处（库/表/连接串）统一 utf8mb4 才彻底（ROADMAP 风险 R4）。
--
-- 表随里程碑增量追加（M1 users → M2 知识库三件套 → RAG完善 会话两件套），
-- 唯一键/索引/外键名与 app/db/models.py 保持一致（uk_*/idx_*/fk_* 前缀），两套 DDL 不许漂移。
-- ============================================================

CREATE DATABASE IF NOT EXISTS zhiyuan
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE zhiyuan;

-- ------------------------------------------------------------
-- 用户表（与 app/db/models.py 的 User 一一对应；M1 交付）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
  id            BIGINT       NOT NULL AUTO_INCREMENT,
  username      VARCHAR(64)  NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_users_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 知识库分组表（M2）：向量库物理隔离的载体——每个 (user_id, group_id)
-- 对应一个独立 Chroma collection，删分组即删整库。
-- 分组名按用户唯一：同一人不能建两个同名分组，不同用户互不影响。
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kb_groups (
  id         BIGINT       NOT NULL AUTO_INCREMENT,
  user_id    BIGINT       NOT NULL,
  name       VARCHAR(128) NOT NULL,
  created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_kb_groups_user_name (user_id, name),
  CONSTRAINT fk_kb_groups_user FOREIGN KEY (user_id)
    REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 文档表（M2）：一个上传文件一行；(user_id, group_id, file_name) 唯一，
-- 同名重传=同一行版本更新（chunk 级增量），保证向量 id "{doc_id}:{chunk_index}" 稳定。
-- empty_pages 存逗号拼接字符串（如 "2,5"）：溯源附属信息，避开 JSON 方言差异。
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
  id          BIGINT       NOT NULL AUTO_INCREMENT,
  user_id     BIGINT       NOT NULL,
  group_id    BIGINT       NOT NULL,
  file_name   VARCHAR(255) NOT NULL,
  file_path   VARCHAR(512) NOT NULL,
  file_hash   VARCHAR(64)  NOT NULL,
  page_count  INT          NOT NULL DEFAULT 0,
  chunk_count INT          NOT NULL DEFAULT 0,
  empty_pages VARCHAR(512) NOT NULL DEFAULT '',
  status      VARCHAR(16)  NOT NULL DEFAULT 'ready',
  created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                            ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_docs_user_group_name (user_id, group_id, file_name),
  CONSTRAINT fk_docs_user FOREIGN KEY (user_id)
    REFERENCES users (id) ON DELETE CASCADE,
  CONSTRAINT fk_docs_group FOREIGN KEY (group_id)
    REFERENCES kb_groups (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 块指纹表（M2）：记录「当前有效」每块的 sha256，chunk 级增量的比对基线。
-- (document_id, chunk_index) 唯一：同一块只留一条当前指纹，
-- replace_chunk_fingerprints 整组替换后指纹表==当前块集合。
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunk_fingerprints (
  id          BIGINT      NOT NULL AUTO_INCREMENT,
  document_id BIGINT      NOT NULL,
  chunk_index INT         NOT NULL,
  chunk_hash  VARCHAR(64) NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uk_chunks_doc_index (document_id, chunk_index),
  CONSTRAINT fk_chunks_document FOREIGN KEY (document_id)
    REFERENCES documents (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 会话表（RAG完善 会话历史）：一次连续答疑一行；updated_at 随消息追加刷新，
-- 列表按 (updated_at DESC, id DESC) 倒序=最近活跃在前。
-- 标题允许重名（无唯一约束）——与 kb_groups 分组名唯一语义不同，会话列表以 id 为键。
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversations (
  id         BIGINT       NOT NULL AUTO_INCREMENT,
  user_id    BIGINT       NOT NULL,
  title      VARCHAR(128) NOT NULL DEFAULT '新对话',
  created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                           ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  CONSTRAINT fk_conversations_user FOREIGN KEY (user_id)
    REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------
-- 消息表（RAG完善 会话历史）：一问一答各一行；hit 可空（用户行 NULL，
-- 助手行 True/False）；sources 存 JSON 字符串（与 empty_pages 同构取舍）。
-- idx_messages_conversation：回看按会话整取消息的查询路径。
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS messages (
  id              BIGINT       NOT NULL AUTO_INCREMENT,
  conversation_id BIGINT       NOT NULL,
  role            VARCHAR(16)  NOT NULL,
  content         TEXT         NOT NULL,
  hit             TINYINT(1)   NULL,
  sources         TEXT         NOT NULL,
  created_at      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_messages_conversation (conversation_id, id),
  CONSTRAINT fk_messages_conversation FOREIGN KEY (conversation_id)
    REFERENCES conversations (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
