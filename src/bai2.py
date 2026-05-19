from pyspark.sql import SparkSession
import os

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("Lab3_Exercise_2")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
sc = spark.sparkContext
sc.setLogLevel("ERROR")

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

num_p = 8


# BƯỚC 1: XỬ LÝ MOVIES -> MovieID: List[Genre]

def parse_movie_genres(line):
    parts = line.split(",")
    movie_id = int(parts[0])
    genres = parts[2].split("|")
    return (movie_id, genres)

movie_genres_map = (
    sc.textFile(os.path.join(BASE, "movies.txt"))
    .map(parse_movie_genres)
    .collectAsMap()
)


# BƯỚC 2: XỬ LÝ RATINGS -> (Genre, (Rating, 1))

def parse_ratings(line):
    parts = line.split(",")
    _, movie_id, rating, _ = parts
    return (int(movie_id), float(rating))

def expand_to_genres(row):
    movie_id, rating = row
    genres = movie_genres_map.get(movie_id, [])
    return [(genre, (rating, 1)) for genre in genres]

ratings_raw = (
    sc.textFile(os.path.join(BASE, "ratings_1.txt"))
    .union(sc.textFile(os.path.join(BASE, "ratings_2.txt")))
)

genre_pairs = (
    ratings_raw
    .map(parse_ratings)
    .flatMap(expand_to_genres)
)


# BƯỚC 3: REDUCE -> tính trung bình theo thể loại

def sum_ratings(a, b):
    return (a[0] + b[0], a[1] + b[1])

def calc_avg(row):
    genre, (total, count) = row
    return (genre, (total / count, count))

genre_avg_rdd = (
    genre_pairs
    .reduceByKey(sum_ratings, numPartitions=num_p)
    .map(calc_avg)
    .sortBy(lambda x: -x[1][0])
    .cache()
)


# BƯỚC 4: GHI KẾT QUẢ

output_path = os.path.join(RESULTS, "bai2_result.txt")
try:
    with open(output_path, "w", encoding="utf-8") as f:
        print("--- ĐIỂM TRUNG BÌNH THEO THỂ LOẠI PHIM ---")
        f.write("--- ĐIỂM TRUNG BÌNH THEO THỂ LOẠI PHIM ---\n")

        for genre, (avg, cnt) in genre_avg_rdd.toLocalIterator():
            line = f"Genre: {genre}, Avg: {avg:.2f}, Count: {cnt}"
            print(line)
            f.write(line + "\n")

    print(f"\nKết quả đã lưu vào: {output_path}")
except Exception as e:
    print(f"Lỗi khi ghi file: {e}")

spark.stop()
