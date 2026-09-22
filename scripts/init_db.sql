-- ============================================================
-- 知源 · MySQL 初始化脚本（幂等：可反复执行不报错）
-- 用法：mysql -u root -p < scripts/init_db.sql
--
-- 为什么统一 utf8mb4：MySQL 8 默认 utf8mb3 存不下 emoji/生僻字且中文易乱码，
-- 建库建表连接三处（库/表/连接串）统一 utf8mb4 才彻底（ROADMAP 风险 R4）。
-- ============================================================

CREATE DATABASE IF NOT EXISTS zhiyuan
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE zhiyuan;

-- 用户表（与 app/db/models.py 的 User 一一对应；M1 阶段仅此一张，
-- 其余业务表随各自里程碑增量建——一个阶段只做一个阶段的事）
CREATE TABLE IF NOT EXISTS users (
  id            BIGINT       NOT NULL AUTO_INCREMENT,
  username      VARCHAR(64)  NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_users_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
