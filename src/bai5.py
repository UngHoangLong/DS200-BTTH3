from pyspark.sql import SparkSession
import os

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("Lab3_Exercise_5")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
sc = spark.sparkContext
sc.setLogLevel("ERROR")

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

num_p = 8


# BƯỚC 1: XỬ LÝ OCCUPATION -> OccID: OccName

def parse_occupation(line):
    parts = line.split(",")
    return (int(parts[0]), parts[1])  # (OccID, OccName)

occupation_map = (
    sc.textFile(os.path.join(BASE, "occupation.txt"))
    .map(parse_occupation)
    .collectAsMap()
)


# BƯỚC 1: XỬ LÝ USERS -> UserID: OccupationName

def parse_users(line):
    parts = line.split(",")
    user_id = int(parts[0])
    occ_name = occupation_map.get(int(parts[3]), "Unknown")
    return (user_id, occ_name)  # (UserID, OccupationName)

user_occ_map = (
    sc.textFile(os.path.join(BASE, "users.txt"))
    .map(parse_users)
    .collectAsMap()
)


# BƯỚC 2: XỬ LÝ RATINGS -> (OccupationName, (Rating, 1))

def parse_ratings_with_occ(line):
    parts = line.split(",")
    user_id, rating = int(parts[0]), float(parts[2])
    occ = user_occ_map.get(user_id, "Unknown")
    return (occ, (rating, 1))

ratings_raw = (
    sc.textFile(os.path.join(BASE, "ratings_1.txt"))
    .union(sc.textFile(os.path.join(BASE, "ratings_2.txt")))
)

occ_pairs = ratings_raw.map(parse_ratings_with_occ)


# BƯỚC 3: REDUCE -> Tính trung bình theo nghề nghiệp

def sum_ratings(a, b):
    return (a[0] + b[0], a[1] + b[1])

def calc_avg(row):
    occ, (total, count) = row
    return (occ, total / count, count)

occ_avg_rdd = (
    occ_pairs
    .reduceByKey(sum_ratings, numPartitions=num_p)
    .map(calc_avg)
    .sortBy(lambda x: -x[1])
    .cache()
)


# BƯỚC 4: GHI KẾT QUẢ

output_path = os.path.join(RESULTS, "bai5_result.txt")
try:
    with open(output_path, "w", encoding="utf-8") as f:
        print("--- ĐIỂM TRUNG BÌNH VÀ SỐ LƯỢT ĐÁNH GIÁ THEO NGHỀ NGHIỆP ---")
        f.write("--- ĐIỂM TRUNG BÌNH VÀ SỐ LƯỢT ĐÁNH GIÁ THEO NGHỀ NGHIỆP ---\n")

        for occ, avg, cnt in occ_avg_rdd.toLocalIterator():
            line = f"Occupation: {occ}, Avg: {avg:.2f}, Count: {cnt}"
            print(line)
            f.write(line + "\n")

    print(f"\nKết quả đã lưu vào: {output_path}")
except Exception as e:
    print(f"Lỗi khi ghi file: {e}")

spark.stop()
