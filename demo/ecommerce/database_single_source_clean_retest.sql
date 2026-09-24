-- Generated SQL 1; call_id=call_00_wlySjtjnRicepppi2JOC5047; executed=true
SELECT MIN(order_date) AS min_dt, MAX(order_date) AS max_dt, COUNT(*) AS cnt, COUNT(DISTINCT substr(order_date,1,7)) AS months FROM orders

-- Generated SQL 2; call_id=call_01_hAxJPUn1Z8D4RwwM61Wk7174; executed=true
SELECT substr(order_date,1,7) AS ym, COUNT(*) AS order_cnt, ROUND(SUM(total_amount),2) AS gmv_all, ROUND(SUM(CASE WHEN order_status IN ('paid','completed') THEN total_amount ELSE 0 END),2) AS gmv_paid_completed FROM orders GROUP BY ym ORDER BY ym

-- Generated SQL 3; call_id=call_02_ELS1MInlpXtiEgLLWlXU9565; executed=true
SELECT COUNT(*) AS aug_cnt, IFNULL(ROUND(SUM(total_amount),2),0) AS aug_gmv FROM orders WHERE order_date >= '2026-08-01' AND order_date < '2026-09-01'

-- Generated SQL 4; call_id=call_00_Vj1pEZ6U0Gqonv0c9cFL4904; executed=true
SELECT order_status, COUNT(*) AS cnt, ROUND(SUM(total_amount),2) AS amt FROM orders WHERE order_date >= '2026-08-01' AND order_date < '2026-09-01' GROUP BY order_status ORDER BY amt DESC

-- Generated SQL 5; call_id=call_01_oNbvxKwBoFoCBBx5H3bw4733; executed=true
SELECT platform, COUNT(*) AS cnt, ROUND(SUM(total_amount),2) AS amt FROM orders WHERE order_date >= '2026-08-01' AND order_date < '2026-09-01' GROUP BY platform ORDER BY amt DESC

-- Generated SQL 6; call_id=call_00_rdKaZDl0NNg3TULVInw08323; executed=false
SELECT COUNT(*) AS item_rows, ROUND(SUM(oi.quantity*oi.unit_price),2) AS gross, ROUND(SUM(oi.discount_amount),2) AS discount, ROUND(SUM(oi.quantity*oi.unit_price - oi.discount_amount),2) AS net, ROUND(SUM(CASE WHEN o.order_status IN ('paid','completed') THEN oi.quantity*oi.unit_price - oi.discount_amount ELSE 0 END),2) AS net_paid_completed FROM order_items oi JOIN orders o ON oi.order_id=o.order_id WHERE o.order_date >= '2026-08-01' AND o.order_date < '2026-09-01'
