from pyspark.sql import SparkSession
import os

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("Lab3_Exercise_3")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
sc = spark.sparkContext
sc.setLogLevel("ERROR")

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

num_p = 8


# BƯỚC 1: XỬ LÝ MOVIES -> MovieID: Title

def parse_movies(line):
    parts = line.split(",")
    return (int(parts[0]), parts[1])  # (MovieID, Title)

movies_rdd = (
    sc.textFile(os.path.join(BASE, "movies.txt"))
    .map(parse_movies)
    .partitionBy(num_p)
    .cache()
)


# BƯỚC 1: XỬ LÝ USERS -> UserID: Gender

def parse_users(line):
    parts = line.split(",")
    return (int(parts[0]), parts[1])  # (UserID, Gender)

user_gender_map = (
    sc.textFile(os.path.join(BASE, "users.txt"))
    .map(parse_users)
    .collectAsMap()
)


# BƯỚC 2: XỬ LÝ RATINGS -> (MovieID, (Gender, Rating))

def parse_ratings_with_gender(line):
    parts = line.split(",")
    user_id, movie_id, rating = int(parts[0]), int(parts[1]), float(parts[2])
    gender = user_gender_map.get(user_id, "Unknown")
    return (movie_id, (gender, rating))

ratings_raw = (
    sc.textFile(os.path.join(BASE, "ratings_1.txt"))
    .union(sc.textFile(os.path.join(BASE, "ratings_2.txt")))
)

# Chuyển thành ((MovieID, Gender), (Rating, 1)) để reduce
def to_key_value(row):
    movie_id, (gender, rating) = row
    return ((movie_id, gender), (rating, 1))

gender_movie_pairs = (
    ratings_raw
    .map(parse_ratings_with_gender)
    .map(to_key_value)
)


# BƯỚC 3: REDUCE -> Tính trung bình

def sum_ratings(a, b):
    return (a[0] + b[0], a[1] + b[1])

def calc_avg(row):
    key, (total, count) = row
    return (key[0], (key[1], total / count, count))  # (MovieID, (Gender, Avg, Count))

# Gom tất cả (Gender, Avg, Count) theo MovieID thành dict
# (MovieID, {Gender: (Avg, Count)})
def calc_avg_keyed(row):
    key, (total, count) = row
    movie_id, gender = key
    avg = total / count
    return (movie_id, {gender: (avg, count)})

def merge_gender_dicts(a, b):
    merged = dict(a)
    merged.update(b)
    return merged

gender_by_movie = (
    gender_movie_pairs
    .reduceByKey(sum_ratings, numPartitions=num_p)
    .map(calc_avg_keyed)
    .reduceByKey(merge_gender_dicts)
    .cache()
)

# JOIN với movies để lấy Title
joined = movies_rdd.join(gender_by_movie).sortBy(lambda x: x[0]).cache()

ALL_GENDERS = ["F", "M"]


# BƯỚC 4: GHI KẾT QUẢ

output_path = os.path.join(RESULTS, "bai3_result.txt")
try:
    with open(output_path, "w", encoding="utf-8") as f:
        header = "--- ĐIỂM TRUNG BÌNH MỖI PHIM THEO GIỚI TÍNH ---"
        print(header)
        f.write(header + "\n")

        for m_id, (title, gender_dict) in joined.toLocalIterator():
            parts = [f"MovieID: {m_id}", f"Title: {title}"]
            for g in ALL_GENDERS:
                if g in gender_dict:
                    avg, cnt = gender_dict[g]
                    parts.append(f"{g}: {avg:.2f}({cnt})")
                else:
                    parts.append(f"{g}: N/A")
            line = ", ".join(parts)
            print(line)
            f.write(line + "\n")

    print(f"\nKết quả đã lưu vào: {output_path}")
except Exception as e:
    print(f"Lỗi khi ghi file: {e}")

spark.stop()
