SELECT family, SUM(sales) AS units, SUM(promo_items) AS promoted_items,
                       AVG(sales) AS average_weekly_units,
                       RANK() OVER (ORDER BY SUM(sales) DESC) AS demand_rank
                FROM weekly_demand GROUP BY family ORDER BY units DESC;

SELECT store_nbr, city, state, type, SUM(sales) AS units,
                       SUM(transactions) / COUNT(DISTINCT family) AS transactions,
                       RANK() OVER (ORDER BY SUM(sales) DESC) AS demand_rank
                FROM weekly_demand GROUP BY store_nbr, city, state, type ORDER BY units DESC;

WITH tagged AS (
                    SELECT *, CASE WHEN promo_items > 0 THEN 'Promoted' ELSE 'Not promoted' END AS status
                    FROM weekly_demand
                )
                SELECT status, COUNT(*) AS observations, AVG(sales) AS average_units,
                       SUM(sales) AS total_units FROM tagged GROUP BY status;

WITH company AS (
                    SELECT week, SUM(sales) AS units FROM weekly_demand GROUP BY week
                )
                SELECT week, units,
                       AVG(units) OVER (ORDER BY week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS four_week_average
                FROM company ORDER BY week;