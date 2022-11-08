import json
from dataprofiler import Data, Profiler

# Auto-Detect & Load: CSV, AVRO, Parquet, JSON, Text
data = Data("data/csv/muckrock-ml-covid.csv")
# Access data directly via a compatible Pandas DataFrame
print(data.data.head(5))
# Calculate Statistics, Entity Recognition, etc
profile = Profiler(data)
readable_report = profile.report(report_options={"output_format": "pretty"})
print(json.dumps(readable_report, indent=4))
