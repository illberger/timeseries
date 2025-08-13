
CREATE TABLE [dbo].[CandleSticks](
	[CandleStickId] [int] IDENTITY(1,1) NOT NULL,
	[Symbol] [nvarchar](20) NULL,
	[Interval] [nvarchar](10) NULL,
	[OpenTime] [bigint] NULL,
	[OpenPrice] [decimal](18, 8) NULL,
	[HighPrice] [decimal](18, 8) NULL,
	[LowPrice] [decimal](18, 8) NULL,
	[ClosePrice] [decimal](18, 8) NULL,
	[Volume] [decimal](18, 8) NULL,
	[CloseTime] [bigint] NULL,
PRIMARY KEY CLUSTERED 
(
	[CandleStickId] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO


