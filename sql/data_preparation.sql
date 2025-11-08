-- =====================================================
-- Customer Segmentation Analysis - Data Preparation
-- SQL queries for extracting and transforming customer data
-- =====================================================

-- Query 1: Customer Base Information
-- Extract core customer attributes
SELECT 
    customer_id,
    first_name,
    last_name,
    email,
    phone,
    date_of_birth,
    gender,
    TIMESTAMPDIFF(YEAR, date_of_birth, CURRENT_DATE) AS age,
    city,
    state,
    country,
    registration_date,
    TIMESTAMPDIFF(MONTH, registration_date, CURRENT_DATE) AS tenure_months
FROM customers
WHERE status = 'active'
  AND registration_date >= '2020-01-01';


-- Query 2: Customer Purchase Behavior
-- Aggregate purchase metrics for each customer
SELECT 
    c.customer_id,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COUNT(DISTINCT DATE(o.order_date)) AS purchase_frequency,
    SUM(o.order_amount) AS total_revenue,
    AVG(o.order_amount) AS avg_transaction_value,
    MAX(o.order_date) AS last_purchase_date,
    MIN(o.order_date) AS first_purchase_date,
    DATEDIFF(CURRENT_DATE, MAX(o.order_date)) AS days_since_last_purchase
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_status = 'completed'
  AND o.order_date >= '2022-01-01'
GROUP BY c.customer_id
HAVING COUNT(DISTINCT o.order_id) >= 1;


-- Query 3: Customer Product Category Preferences
-- Identify primary product categories for each customer
SELECT 
    c.customer_id,
    pc.category_name AS primary_category,
    COUNT(DISTINCT oi.product_id) AS unique_products_purchased,
    SUM(oi.quantity) AS total_items_purchased,
    SUM(oi.line_total) AS category_revenue
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
INNER JOIN order_items oi ON o.order_id = oi.order_id
INNER JOIN products p ON oi.product_id = p.product_id
INNER JOIN product_categories pc ON p.category_id = pc.category_id
WHERE o.order_status = 'completed'
GROUP BY c.customer_id, pc.category_name
ORDER BY c.customer_id, category_revenue DESC;


-- Query 4: Customer Lifetime Value (CLV) Calculation
-- Calculate estimated customer lifetime value
WITH customer_metrics AS (
    SELECT 
        customer_id,
        COUNT(DISTINCT order_id) AS order_count,
        SUM(order_amount) AS total_spent,
        AVG(order_amount) AS avg_order_value,
        DATEDIFF(MAX(order_date), MIN(order_date)) AS customer_lifespan_days,
        MIN(order_date) AS first_order_date,
        MAX(order_date) AS last_order_date
    FROM orders
    WHERE order_status = 'completed'
    GROUP BY customer_id
)
SELECT 
    customer_id,
    order_count,
    total_spent,
    avg_order_value,
    customer_lifespan_days,
    CASE 
        WHEN customer_lifespan_days > 0 THEN 
            (total_spent / customer_lifespan_days) * 365
        ELSE total_spent
    END AS estimated_annual_value,
    CASE 
        WHEN customer_lifespan_days > 0 THEN 
            (total_spent / customer_lifespan_days) * 365 * 3
        ELSE total_spent * 3
    END AS estimated_3year_clv
FROM customer_metrics
WHERE order_count >= 2;


-- Query 5: Customer Engagement Score
-- Calculate engagement score based on multiple factors
WITH engagement_factors AS (
    SELECT 
        c.customer_id,
        COUNT(DISTINCT o.order_id) AS purchase_count,
        COUNT(DISTINCT r.review_id) AS review_count,
        COUNT(DISTINCT w.item_id) AS wishlist_items,
        SUM(CASE WHEN o.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAYS) 
            THEN 1 ELSE 0 END) AS recent_purchases_90d,
        DATEDIFF(CURRENT_DATE, MAX(o.order_date)) AS recency_days
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    LEFT JOIN reviews r ON c.customer_id = r.customer_id
    LEFT JOIN wishlist w ON c.customer_id = w.customer_id
    GROUP BY c.customer_id
)
SELECT 
    customer_id,
    purchase_count,
    review_count,
    wishlist_items,
    recent_purchases_90d,
    recency_days,
    -- Engagement score (0-100 scale)
    LEAST(100, 
        (purchase_count * 5) + 
        (review_count * 3) + 
        (wishlist_items * 1) + 
        (recent_purchases_90d * 10) - 
        (recency_days * 0.1)
    ) AS engagement_score
FROM engagement_factors;


-- Query 6: RFM Analysis (Recency, Frequency, Monetary)
-- Classic RFM segmentation preparation
WITH rfm_base AS (
    SELECT 
        customer_id,
        MAX(order_date) AS last_purchase_date,
        COUNT(DISTINCT order_id) AS frequency,
        SUM(order_amount) AS monetary_value
    FROM orders
    WHERE order_status = 'completed'
      AND order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 2 YEAR)
    GROUP BY customer_id
)
SELECT 
    customer_id,
    DATEDIFF(CURRENT_DATE, last_purchase_date) AS recency_days,
    frequency,
    monetary_value,
    -- RFM Scores (1-5 scale)
    NTILE(5) OVER (ORDER BY DATEDIFF(CURRENT_DATE, last_purchase_date) DESC) AS recency_score,
    NTILE(5) OVER (ORDER BY frequency ASC) AS frequency_score,
    NTILE(5) OVER (ORDER BY monetary_value ASC) AS monetary_score
FROM rfm_base;


-- Query 7: Final Integrated Customer Dataset
-- Combine all metrics for segmentation analysis
CREATE TABLE customer_segmentation_dataset AS
SELECT 
    c.customer_id,
    TIMESTAMPDIFF(YEAR, c.date_of_birth, CURRENT_DATE) AS age,
    c.gender,
    c.city,
    c.state,
    TIMESTAMPDIFF(MONTH, c.registration_date, CURRENT_DATE) AS tenure_months,
    COALESCE(pm.total_orders, 0) AS purchase_frequency,
    COALESCE(pm.avg_transaction_value, 0) AS avg_transaction_value,
    COALESCE(pm.total_revenue, 0) AS total_revenue,
    COALESCE(pm.days_since_last_purchase, 999) AS days_since_last_purchase,
    COALESCE(es.engagement_score, 0) AS engagement_score,
    COALESCE(rfm.recency_score, 1) AS recency_score,
    COALESCE(rfm.frequency_score, 1) AS frequency_score,
    COALESCE(rfm.monetary_score, 1) AS monetary_score,
    -- Derived spending score
    CASE 
        WHEN pm.total_revenue > 5000 THEN 'High'
        WHEN pm.total_revenue > 2000 THEN 'Medium'
        ELSE 'Low'
    END AS spending_category
FROM customers c
LEFT JOIN (
    SELECT 
        customer_id,
        COUNT(DISTINCT order_id) AS total_orders,
        AVG(order_amount) AS avg_transaction_value,
        SUM(order_amount) AS total_revenue,
        DATEDIFF(CURRENT_DATE, MAX(order_date)) AS days_since_last_purchase
    FROM orders
    WHERE order_status = 'completed'
    GROUP BY customer_id
) pm ON c.customer_id = pm.customer_id
LEFT JOIN (
    -- Engagement score subquery
    SELECT customer_id, engagement_score
    FROM engagement_factors
) es ON c.customer_id = es.customer_id
LEFT JOIN (
    -- RFM scores subquery
    SELECT customer_id, recency_score, frequency_score, monetary_score
    FROM rfm_analysis
) rfm ON c.customer_id = rfm.customer_id
WHERE c.status = 'active';


-- Query 8: Export Dataset for Python Analysis
-- Final query to export clean dataset
SELECT 
    customer_id,
    age,
    gender,
    tenure_months,
    purchase_frequency,
    avg_transaction_value,
    total_revenue,
    engagement_score,
    recency_score,
    frequency_score,
    monetary_score,
    spending_category
FROM customer_segmentation_dataset
WHERE age BETWEEN 18 AND 80
  AND tenure_months >= 1
ORDER BY customer_id;

-- =====================================================
-- End of Data Preparation Queries
-- =====================================================
