from pyspark.sql import SparkSession
from datetime import datetime
import os

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("Lab3_Exercise_6")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
sc = spark.sparkContext
sc.setLogLevel("ERROR")

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

num_p = 8


# BƯỚC 1: XỬ LÝ RATINGS -> (Year, (Rating, 1))

def parse_ratings_with_year(line):
    parts = line.split(",")
    rating = float(parts[2])
    year = datetime.utcfromtimestamp(int(parts[3])).year
    return (year, (rating, 1))

ratings_raw = (
    sc.textFile(os.path.join(BASE, "ratings_1.txt"))
    .union(sc.textFile(os.path.join(BASE, "ratings_2.txt")))
)

year_pairs = ratings_raw.map(parse_ratings_with_year)


# BƯỚC 2: REDUCE -> Tính tổng lượt và trung bình theo năm

def sum_ratings(a, b):
    return (a[0] + b[0], a[1] + b[1])

def calc_avg(row):
    year, (total, count) = row
    return (year, total / count, count)

year_avg_rdd = (
    year_pairs
    .reduceByKey(sum_ratings, numPartitions=num_p)
    .map(calc_avg)
    .sortBy(lambda x: x[0])
    .cache()
)


# BƯỚC 3: GHI KẾT QUẢ

output_path = os.path.join(RESULTS, "bai6_result.txt")
try:
    with open(output_path, "w", encoding="utf-8") as f:
        print("--- TỔNG LƯỢT ĐÁNH GIÁ VÀ ĐIỂM TRUNG BÌNH THEO NĂM ---")
        f.write("--- TỔNG LƯỢT ĐÁNH GIÁ VÀ ĐIỂM TRUNG BÌNH THEO NĂM ---\n")

        for year, avg, cnt in year_avg_rdd.toLocalIterator():
            line = f"Year: {year}, Avg: {avg:.2f}, Count: {cnt}"
            print(line)
            f.write(line + "\n")

    print(f"\nKết quả đã lưu vào: {output_path}")
except Exception as e:
    print(f"Lỗi khi ghi file: {e}")

spark.stop()
