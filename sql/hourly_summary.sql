-- Historical description of TRAINING data, not a forecasting model.
-- Notebook 01 independently calculates the same result with pandas.
SELECT workingday,
       hr,
       COUNT(*) AS observed_hours,
       SUM(cnt) AS total_rentals,
       AVG(cnt) AS mean_rentals
FROM hourly
GROUP BY workingday, hr
ORDER BY workingday, hr;
