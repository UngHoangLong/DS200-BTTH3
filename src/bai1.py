from pyspark.sql import SparkSession
import os

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("Lab3_Exercise_1")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
sc = spark.sparkContext
sc.setLogLevel("ERROR")

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

num_p = 8


# BƯỚC 1: XỬ LÝ MOVIES (MAP)

def parse_movies(line):
    parts = line.split(",")
    return (int(parts[0]), parts[1])  # (MovieID, Title)

movies_rdd = (
    sc.textFile(os.path.join(BASE, "movies.txt"))
    .map(parse_movies)
    .partitionBy(num_p)
    .cache()
)


# BƯỚC 2: XỬ LÝ RATINGS (UNION -> MAP -> REDUCE)

def parse_ratings(line):
    parts = line.split(",")
    _, movie_id, rating, _ = parts
    return (int(movie_id), (float(rating), 1))  # (MovieID, (Rating, 1))

def sum_ratings(a, b):
    return (a[0] + b[0], a[1] + b[1])

def calc_avg(row):
    movie_id, (total_score, total_count) = row
    return (movie_id, (total_score / total_count, total_count))

ratings_raw = (
    sc.textFile(os.path.join(BASE, "ratings_1.txt"))
    .union(sc.textFile(os.path.join(BASE, "ratings_2.txt")))
)
ratings_final = (
    ratings_raw
    .map(parse_ratings)
    .reduceByKey(sum_ratings, numPartitions=num_p)
    .map(calc_avg)
)


# BƯỚC 3: JOIN movies + ratings

# (MovieID, (Title, (AvgRating, Count)))
movie_rating_rdd = movies_rdd.join(ratings_final).cache()


# BƯỚC 4: GHI KẾT QUẢ

output_path = os.path.join(RESULTS, "bai1_result.txt")
try:
    with open(output_path, "w", encoding="utf-8") as f:
        print("--- DANH SÁCH PHIM - ĐIỂM TRUNG BÌNH - SỐ LƯỢT ĐÁNH GIÁ ---")
        f.write("--- DANH SÁCH PHIM - ĐIỂM TRUNG BÌNH - SỐ LƯỢT ĐÁNH GIÁ ---\n")

        for m_id, (title, (avg, cnt)) in movie_rating_rdd.toLocalIterator():
            line = f"MovieID: {m_id}, Title: {title}, Avg: {avg:.2f}, Count: {cnt}"
            print(line)
            f.write(line + "\n")

        # Lọc phim có ít nhất 5 lượt đánh giá, tìm điểm cao nhất
        filtered = movie_rating_rdd.filter(lambda x: x[1][1][1] >= 5)
        if not filtered.isEmpty():
            top = filtered.max(key=lambda x: x[1][1][0])
            print("\n--- PHIM CÓ ĐIỂM TRUNG BÌNH CAO NHẤT (>= 5 LƯỢT ĐÁNH GIÁ) ---")
            print(f"Title: {top[1][0]}, Avg: {top[1][1][0]:.2f}, Count: {top[1][1][1]}")
            f.write("\n--- PHIM CÓ ĐIỂM TRUNG BÌNH CAO NHẤT (>= 5 LƯỢT ĐÁNH GIÁ) ---\n")
            f.write(f"Title: {top[1][0]}, Avg: {top[1][1][0]:.2f}, Count: {top[1][1][1]}\n")
        else:
            msg = "Không có phim nào có >= 5 lượt đánh giá."
            print(msg)
            f.write(msg + "\n")

    print(f"\nKết quả đã lưu vào: {output_path}")
except Exception as e:
    print(f"Lỗi khi ghi file: {e}")

spark.stop()
