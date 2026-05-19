from pyspark.sql import SparkSession
import os

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("Lab3_Exercise_4")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
sc = spark.sparkContext
sc.setLogLevel("ERROR")

BASE = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

num_p = 8


def get_age_group(age):
    age = int(age)
    if age < 18:
        return "Under 18"
    elif age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    else:
        return "55+"


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


# BƯỚC 1: XỬ LÝ USERS -> UserID: AgeGroup

def parse_users(line):
    parts = line.split(",")
    return (int(parts[0]), get_age_group(parts[2]))  # (UserID, AgeGroup)

user_age_map = (
    sc.textFile(os.path.join(BASE, "users.txt"))
    .map(parse_users)
    .collectAsMap()
)


# BƯỚC 2: XỬ LÝ RATINGS -> ((MovieID, AgeGroup), (Rating, 1))

def parse_ratings_with_age(line):
    parts = line.split(",")
    user_id, movie_id, rating = int(parts[0]), int(parts[1]), float(parts[2])
    age_group = user_age_map.get(user_id, "Unknown")
    return ((movie_id, age_group), (rating, 1))

ratings_raw = (
    sc.textFile(os.path.join(BASE, "ratings_1.txt"))
    .union(sc.textFile(os.path.join(BASE, "ratings_2.txt")))
)

age_movie_pairs = ratings_raw.map(parse_ratings_with_age)


# BƯỚC 3: REDUCE -> Tính trung bình

def sum_ratings(a, b):
    return (a[0] + b[0], a[1] + b[1])

def calc_avg(row):
    key, (total, count) = row
    return (key[0], (key[1], total / count, count))  # (MovieID, (AgeGroup, Avg, Count))

# Gom tất cả (AgeGroup, Avg, Count) theo MovieID thành dict
# (MovieID, {AgeGroup: (Avg, Count)})
def calc_avg_keyed(row):
    key, (total, count) = row
    movie_id, age_group = key
    avg = total / count
    return (movie_id, {age_group: (avg, count)})

def merge_age_dicts(a, b):
    merged = dict(a)
    merged.update(b)
    return merged

age_by_movie = (
    age_movie_pairs
    .reduceByKey(sum_ratings, numPartitions=num_p)
    .map(calc_avg_keyed)
    .reduceByKey(merge_age_dicts)
    .cache()
)

# JOIN với movies để lấy Title
joined = movies_rdd.join(age_by_movie).sortBy(lambda x: x[0]).cache()

ALL_AGE_GROUPS = ["Under 18", "18-24", "25-34", "35-44", "45-54", "55+"]


# BƯỚC 4: GHI KẾT QUẢ

output_path = os.path.join(RESULTS, "bai4_result.txt")
try:
    with open(output_path, "w", encoding="utf-8") as f:
        header = "--- ĐIỂM TRUNG BÌNH MỖI PHIM THEO NHÓM TUỔI ---"
        print(header)
        f.write(header + "\n")

        for m_id, (title, age_dict) in joined.toLocalIterator():
            parts = [f"MovieID: {m_id}", f"Title: {title}"]
            for g in ALL_AGE_GROUPS:
                if g in age_dict:
                    avg, cnt = age_dict[g]
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
