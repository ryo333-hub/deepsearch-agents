-- Synthetic demo only. Explicit management import, NOT a Docker init replacement.
-- Password is a bound parameter supplied by seed_ecommerce_demo.py; never literal.
-- Existing schemas/accounts cause the importer to stop. No destructive reset.
CREATE DATABASE insight_ecommerce_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE insight_ecommerce_db;
CREATE TABLE products (
 product_id INT PRIMARY KEY,
 product_name VARCHAR(100) NOT NULL,
 category VARCHAR(50) NOT NULL,
 cost_price DECIMAL(12,2) NOT NULL CHECK (cost_price >= 0),
 sale_price DECIMAL(12,2) NOT NULL CHECK (sale_price >= 0),
 status VARCHAR(20) NOT NULL CHECK (status IN ('active','inactive')),
 created_at DATETIME NOT NULL
);
CREATE TABLE orders (
 order_id BIGINT PRIMARY KEY,
 customer_id VARCHAR(32) NOT NULL,
 order_date DATETIME NOT NULL,
 paid_at DATETIME NULL COMMENT 'Business time Asia/Shanghai, use for GMV window',
 platform VARCHAR(20) NOT NULL,
 order_status VARCHAR(20) NOT NULL,
 total_amount DECIMAL(14,2) NOT NULL CHECK (total_amount >= 0),
 CHECK ((order_status IN ('paid','completed') AND paid_at IS NOT NULL AND paid_at >= order_date)
     OR (order_status IN ('cancelled','unpaid') AND paid_at IS NULL)),
 INDEX ix_orders_paid (paid_at), INDEX ix_orders_status (order_status), INDEX ix_orders_platform (platform)
);
CREATE TABLE order_items (
 order_item_id BIGINT PRIMARY KEY,
 order_id BIGINT NOT NULL,
 product_id INT NOT NULL,
 quantity INT NOT NULL CHECK (quantity > 0),
 unit_price DECIMAL(12,2) NOT NULL CHECK (unit_price >= 0),
 discount_amount DECIMAL(12,2) NOT NULL COMMENT 'Discount for the entire line',
 unit_cost_snapshot DECIMAL(12,2) NOT NULL CHECK (unit_cost_snapshot >= 0),
 CHECK (discount_amount >= 0 AND discount_amount <= quantity * unit_price),
 FOREIGN KEY (order_id) REFERENCES orders(order_id),
 FOREIGN KEY (product_id) REFERENCES products(product_id)
);
CREATE TABLE inventory (
 inventory_id INT PRIMARY KEY,
 product_id INT NOT NULL,
 stock_quantity INT NOT NULL CHECK (stock_quantity >= 0),
 safety_stock INT NOT NULL CHECK (safety_stock >= 0),
 warehouse VARCHAR(50) NOT NULL,
 updated_at DATETIME NOT NULL COMMENT '2026-09-01 available stock snapshot',
 FOREIGN KEY (product_id) REFERENCES products(product_id),
 UNIQUE KEY uq_inventory_product_warehouse (product_id, warehouse)
);
CREATE TABLE ad_metrics (
 metric_id BIGINT PRIMARY KEY,
 metric_date DATE NOT NULL,
 platform VARCHAR(20) NOT NULL,
 campaign_name VARCHAR(100) NOT NULL,
 product_id INT NOT NULL,
 impressions BIGINT NOT NULL CHECK (impressions >= 0),
 clicks INT NOT NULL CHECK (clicks >= 0),
 ad_spend DECIMAL(14,2) NOT NULL CHECK (ad_spend >= 0),
 conversions INT NOT NULL CHECK (conversions >= 0),
 revenue DECIMAL(14,2) NOT NULL COMMENT 'Attributed line revenue, not total store GMV' CHECK (revenue >= 0),
 CHECK (clicks <= impressions),
 CHECK (conversions <= clicks),
 FOREIGN KEY (product_id) REFERENCES products(product_id),
 UNIQUE KEY uq_ad_grain (metric_date, platform, campaign_name, product_id),
 INDEX ix_ad_platform (platform)
);
-- FK indexes and leftmost prefixes already cover order_id/product_id/date.
CREATE USER 'ecommerce_ro'@'%' IDENTIFIED BY %s;
-- Escape underscores: database-level grants can interpret them as wildcards.
GRANT SELECT ON `insight\_ecommerce\_db`.* TO 'ecommerce_ro'@'%';
