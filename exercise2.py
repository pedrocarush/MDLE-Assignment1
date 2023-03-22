import os.path
import random
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Row, DataFrame
from pyspark.sql.types import StringType, ArrayType, IntegerType, StructType, StructField
from itertools import combinations, chain
from functools import partial
from typing import Iterable, Any, List, Callable

from argparse import ArgumentParser

"""
Python-file version of the Jupyter notebook, for submission via spark-submit.
The documentation is present in the notebook.
"""



def initialize_spark() -> SparkSession:
    return SparkSession.builder \
        .appName('LSH') \
        .config('spark.master', 'local[*]') \
        .getOrCreate()


def prepare_data(spark: SparkSession, path: str, partitions: int) -> DataFrame:
    return spark.read \
        .json(path) \
        .repartition(partitions)


def generate_shingles(df: DataFrame, shingle_size: int, partitions: int):

    @F.udf(returnType=ArrayType(IntegerType(), False))
    def generate_shingles(text: str):
        shingles = (text[idx:idx+shingle_size] for idx in range(len(text) - shingle_size + 1))
        # Get last 32 bits in order to have 4-byte integers (Python allows arbitrarily large integers)
        to_integer = lambda s: hash(s) & ((1 << 32) - 1)
        return list(set(to_integer(shingle_str) for shingle_str in shingles))

    return df \
        .drop('url') \
        .filter(F.length('text') >= shingle_size) \
        .repartition(partitions) \
        .withColumn('shingles', generate_shingles('text')) \
        .drop('text')


def calculate_min_hash(spark: SparkSession, df_shingles: DataFrame, b: int, r: int):

    # Assumes the values to hash are 4-byte integers
    def generate_universal_hash_family(K: int) -> List[Callable[[int], int]]:
        N = 1 << 32
        p = 2305843009213693951

        parameters = set()
        while (len(parameters) < K):
            parameters |= {(random.randint(1, N), random.randint(0, N)) for _ in range(K - len(parameters))}
        
        return [(a, b, p, N) for a, b in parameters]

    hash_family = generate_universal_hash_family(r * b)
    broadcasted_hash_family = spark.sparkContext.broadcast(hash_family)

    @F.udf(returnType=ArrayType(IntegerType(), False))
    def calculate_min_hash(shingles: List[int]):
        return [min(((a * shingle + b) % p) % N for shingle in shingles) for (a, b, p, N) in broadcasted_hash_family.value]

    return df_shingles.withColumn('min_hash', calculate_min_hash('shingles')).drop('shingles')


def generate_candidate_pairs(spark: SparkSession, df_minhash: DataFrame, b: int, r: int):
    
    @F.udf(returnType=ArrayType(ArrayType(IntegerType(), False), False))
    def generate_even_slices(minhashes: List[int]):
        return [minhashes[i:i+b] for i in range(0, b * r, b)]
    
    df_bands = df_minhash \
        .withColumn('min_hash_slices', generate_even_slices('min_hash')) \
        .select('tweet_id', *(F.hash(F.col('min_hash_slices')[band]).alias(f'band_{band}') for band in range(b)))

    @F.udf(returnType=ArrayType(ArrayType(StringType(), False), False))
    def combine_pairs(elems: Iterable[Any]):
        return list(combinations(elems, 2))

    df_bands_lst = [
        df_bands
            .select('tweet_id', f'band_{band}')
            .groupby(f'band_{band}')
            .agg(F.collect_list('tweet_id'))
            .withColumnRenamed('collect_list(tweet_id)', 'candidates')
            .withColumn('candidates', F.array_sort('candidates'))
            .select(F.explode(combine_pairs('candidates')).alias('candidate_pair'))
        for band in range(b)
    ]

    df_candidate_pairs = spark.createDataFrame([], schema=StructType([StructField(name='candidate_pair', dataType=ArrayType(StringType(), False), nullable=False)]))

    for d in df_bands_lst:
        df_candidate_pairs = df_candidate_pairs.union(d)

    return df_candidate_pairs.distinct()


def filter_false_positives(df_candidate_pairs: DataFrame, df_minhash: DataFrame, similarity_threshold: float) -> DataFrame:
    return df_candidate_pairs \
        .join(df_minhash, df_minhash['tweet_id'] == F.col('candidate_pair')[0]) \
        .withColumnRenamed('min_hash', 'min_hash_first') \
        .drop('tweet_id') \
        .join(df_minhash, df_minhash['tweet_id'] == F.col('candidate_pair')[1]) \
        .withColumnRenamed('min_hash', 'min_hash_second') \
        .drop('tweet_id') \
        .withColumn('similarity', F.size(F.array_intersect('min_hash_first', 'min_hash_second')) / F.size(F.array_union('min_hash_first', 'min_hash_second'))) \
        .filter(F.col('similarity') >= similarity_threshold)


def save_and_load_df(spark: SparkSession, df: DataFrame, fname: str) -> DataFrame:
    if not os.path.exists(fname):
        df.write.mode('overwrite').parquet(path=fname, compression='gzip')

    return spark.read.parquet(fname)


def get_similar_articles(df_candidate_pairs: DataFrame, tweet_id: str) -> List[str]:
    rows = df_candidate_pairs \
        .filter(F.array_contains('candidate_pair', tweet_id)) \
        .select(F.array_remove('candidate_pair', tweet_id).alias('sole_candidate')) \
        .select(F.col('sole_candidate')[0].alias('similar_article')) \
        .collect()

    return [row.similar_article for row in rows]



def main(
        dataset: str,
        shingle_size: int,
        bands: int,
        rows: int,
        seed: int,
        partitions: int,
        similar_to: str=None,
        fpfn_analysis: bool=False,
        fpfn_analysis_samples: int=1,
        fpfn_analysis_fraction: float=0.01
):
    
    random.seed(seed)

    spark = initialize_spark()
    
    if similar_to is not None:
        filter_false_positives(df_candidate_pairs, )

    df = prepare_data(spark, dataset, partitions)

    df_shingles = generate_shingles(df, shingle_size)

    df_minhash = calculate_min_hash(spark, df_shingles, rows, bands)
    df_minhash = save_and_load_df(spark, df_minhash, f'minhash_{rows}_{bands}')

    df_candidate_pairs = generate_candidate_pairs(spark, df_minhash, bands, rows)

    if fpfn_analysis:
        fpfn_analyze(fpfn_analysis_samples, fpfn_analysis_fraction)

    

    spark.stop()



if __name__ == '__main__':

    default_str = ' (default: %(default)s)'

    parser = ArgumentParser()
    parser.add_argument("file", type=str, help="path to the input dataset")
    parser.add_argument("-k", type=int, default=9, help="shingle size" + default_str)
    parser.add_argument("-b", "--bands", type=int, default=13, help="number of LSH bands" + default_str)
    parser.add_argument("-r", "--rows", type=int, default=11, help="number of LSH rows" + default_str)
    parser.add_argument("-s", "--seed", type=int, default=1, help="seed for the random number generator" + default_str)
    parser.add_argument("-p", "--partitions", type=int, default=8, help="number of partitions (mainly for filtering operations)")
    parser.add_argument("--similar-to", type=str, default=None, help="if set, return the most similar tweets to the provided tweet" + default_str)
    parser.add_argument("--fpfn-analysis", action="store_true", help="perform analysis on FP (false positives) and FN (false negatives) on samples of the dataset")
    parser.add_argument("--fpfn-analysis-samples", type=int, default=1, help="number of samples for the FP/FN analysis" + default_str)
    parser.add_argument("--fpfn-analysis-fraction", type=float, default=0.01, help="fraction of the dataset to use for each sample for the FP/FN analysis" + default_str)
    
    args = parser.parse_args()

    main(
        dataset=args.file,
        shingle_size=args.k,
        bands=args.bands,
        rows=args.rows,
        seed=args.seed,
        partitions=args.partitions,
        similar_to=args.similar_to,
        fpfn_analysis=args.fpfn_analysis,
        fpfn_analysis_samples=args.fpfn_analysis_samples,
        fpfn_analysis_fraction=args.fpfn_analysis_fraction
    )
