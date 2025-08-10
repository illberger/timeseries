
-- Query to optimize SQL streaming during Training of NN, but not obligatory.
-- Adds an extra column "PriceTimeMs" to avoid DBMS overhead (/fetching/polymarket_fetcher.py inserts a datetime type)

ALTER TABLE dbo.Polymarket
  ADD PriceTimeMs BIGINT NULL;
GO

UPDATE dbo.Polymarket
SET PriceTimeMs = DATEDIFF_BIG(
    ms,
    '1970-01-01',
    DATEADD(hour, -2, PriceTime)
);
GO

ALTER TABLE dbo.Polymarket
  ALTER COLUMN PriceTimeMs BIGINT NOT NULL;
GO

CREATE INDEX IX_Polymarket_PriceTimeMs
  ON dbo.Polymarket(PriceTimeMs);
GO



