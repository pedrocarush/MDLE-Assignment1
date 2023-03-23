import os.path
import random
import math
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Row, DataFrame
from pyspark.sql.types import StringType, ArrayType, IntegerType, StructType, StructField
from itertools import combinations, chain
from functools import partial
from typing import Iterable, Any, List, Callable

from argparse import ArgumentParser

"""
Python-file version of Exercise 2's Jupyter notebook, for submission via spark-submit.
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


def generate_candidate_pairs(spark: SparkSession, df_minhash: DataFrame, b: int, r: int, partitions: int):
    
    @F.udf(returnType=ArrayType(ArrayType(IntegerType(), False), False))
    def generate_even_slices(minhashes: List[int]):
        return [minhashes[i:i+r] for i in range(0, b * r, r)]
    
    df_bands = df_minhash \
        .withColumn('min_hash_slices', generate_even_slices('min_hash')) \
        .withColumn('bands', F.array(*(
            F.struct(
                F.hash(F.col('min_hash_slices')[band]).alias('band_hash'),
                F.lit(band).alias('band')
            )
            for band in range(b))
        )) \
        .withColumn('bands', F.explode('bands')) \
        .select('tweet_id', F.col('bands').band.alias('band'), F.col('bands').band_hash.alias('band_hash'))

    @F.udf(returnType=ArrayType(ArrayType(StringType(), False), False))
    def combine_pairs(elems: Iterable[Any]):
        return list(combinations(elems, 2))

    df_candidate_pairs = df_bands \
        .groupby('band', 'band_hash') \
        .agg(F.collect_list('tweet_id')) \
        .withColumnRenamed('collect_list(tweet_id)', 'candidates') \
        .withColumn('candidates', F.array_sort('candidates')) \
        .filter(F.size('candidates') > 1) \
        .repartition(partitions) \
        .select('candidates') \
        .distinct() \
        .select(F.explode(combine_pairs('candidates')).alias('candidate_pair')) \
        .select(F.col('candidate_pair')[0].alias('candidate_pair_first'), F.col('candidate_pair')[1].alias('candidate_pair_second')) \
        .distinct() 

    return df_candidate_pairs


@F.udf(returnType=IntegerType())
def min_hash_similar(min_hash_0: List[int], min_hash_1: List[int]):
    return sum((elem0 == elem1) for elem0, elem1 in zip(min_hash_0, min_hash_1))

def filter_false_positives(df_candidate_pairs: DataFrame, df_minhash: DataFrame, similarity_threshold: float, b: int, r: int) -> DataFrame:
    return df_candidate_pairs \
        .join(df_minhash, df_minhash['tweet_id'] == F.col('candidate_pair_first')) \
        .withColumnRenamed('min_hash', 'min_hash_first') \
        .drop('tweet_id') \
        .join(df_minhash, df_minhash['tweet_id'] == F.col('candidate_pair_second')) \
        .withColumnRenamed('min_hash', 'min_hash_second') \
        .drop('tweet_id') \
        .withColumn('similarity', min_hash_similar('min_hash_first', 'min_hash_second') / (b * r)) \
        .filter(F.col('similarity') >= similarity_threshold)


def save_and_load_df(spark: SparkSession, df: DataFrame, fname: str) -> DataFrame:
    if not os.path.exists(fname):
        df.write.mode('overwrite').parquet(path=fname, compression='gzip')

    return spark.read.parquet(fname)


def get_similar_articles(df_candidate_pairs: DataFrame, tweet_id: str) -> List[str]:
    rows = df_candidate_pairs \
        .withColumn('candidate_pair_first', F.when(F.col('candidate_pair_first') != tweet_id, F.col('candidate_pair_first'))) \
        .withColumn('candidate_pair_second', F.when(F.col('candidate_pair_second') != tweet_id, F.col('candidate_pair_second'))) \
        .filter(F.col('candidate_pair_first').isNull() | F.col('candidate_pair_second').isNull()) \
        .select(F.coalesce('candidate_pair_first', 'candidate_pair_second').alias('similar_article')) \
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
        fpfn_analysis_fraction: float=0.01,
        minhash_base: str='minhash',
        candidate_pairs_base: str='candidate_pairs',
):
    
    random.seed(seed)

    similarity_threshold = (1 - bands)**(1 - rows)

    spark = initialize_spark()
    spark.sparkContext.setLogLevel('ERROR')
    
    df = prepare_data(spark, dataset, partitions)

    df_shingles = generate_shingles(df, shingle_size, partitions)

    df_minhash = calculate_min_hash(spark, df_shingles, rows, bands)
    print('Saving the minhashes...', end=' ')
    df_minhash = save_and_load_df(spark, df_minhash, f'{minhash_base}_{rows}_{bands}')
    print('and loaded')

    df_candidate_pairs = generate_candidate_pairs(spark, df_minhash, bands, rows, partitions)
    df_candidate_pairs_fpless = filter_false_positives(df_candidate_pairs, df_minhash, similarity_threshold, bands, rows)
    print('Saving the candidate pairs... ', end='')
    df_candidate_pairs_fpless = save_and_load_df(spark, df_candidate_pairs_fpless, f'{candidate_pairs_base}_{rows}_{bands}')
    print('and loaded')

    if similar_to is not None:
        print('Getting articles...', end=' ')
        similar_articles = get_similar_articles(df_candidate_pairs_fpless, similar_to)
        print(f'similar to \'{similar_to}\':')
        print(*similar_articles, sep='\n')

    if fpfn_analysis:
        print('Analyzing false positives/negatives')
        false_positive_percentages = []
        false_negative_percentages = []
        
        for sample_i in range(fpfn_analysis_samples):
            print(f'Calculating sample number {sample_i + 1}...', end=' ')
            
            df_minhash_sample = df_minhash.sample(fraction=fpfn_analysis_fraction, seed=seed, withReplacement=False)
            
            df_candidate_pairs_sample = generate_candidate_pairs(spark, df_minhash, bands, rows, partitions)

            df_candidate_pairs_fpless_sample = filter_false_positives(df_candidate_pairs_sample, df_minhash_sample, similarity_threshold, bands, rows)

            print('minhashes and candidate pairs calculated...', end=' ')

            candidate_pairs_n = df_candidate_pairs_sample.count()
            candidate_pairs_fpless_n = df_candidate_pairs_fpless_sample.count()
            false_positive_percentage = (candidate_pairs_n - candidate_pairs_fpless_n) / candidate_pairs_n

            false_negatives = df_minhash_sample \
                .crossJoin(df_minhash_sample.select(F.col('tweet_id').alias('tweet_id_other'), F.col('min_hash').alias('min_hash_other'))) \
                .filter(F.col('tweet_id') < F.col('tweet_id_other')) \
                .select(F.array('tweet_id', 'tweet_id_other').alias('pair'),'min_hash', 'min_hash_other') \
                .join(df_candidate_pairs_sample, F.array(df_candidate_pairs_sample['candidate_pair_first'], df_candidate_pairs_sample['candidate_pair_second']) == F.col('pair'), 'left') \
                .filter(F.col('candidate_pair_first').isNull()) \
                .drop('candidate_pair_first', 'candidate_pair_second') \
                .withColumn('similarity', min_hash_similar('min_hash', 'min_hash_other') / (bands * rows)) \
                .filter(F.col('similarity') >= similarity_threshold) \
                .count()

            tweets_n = df_minhash_sample.count()
            false_negative_percentage = false_negatives / (math.comb(tweets_n, 2) - candidate_pairs_n)

            print('false positive and negative percentage calculated')

            false_positive_percentages.append(false_positive_percentage)
            false_negative_percentages.append(false_negative_percentage)
        
        avg = lambda l: sum(l)/len(l)
        print(f'False positive percentage (average over {fpfn_analysis_samples} of fraction {fpfn_analysis_fraction}): {avg(false_positive_percentages):%}')
        print(f'False positive percentage (average over {fpfn_analysis_samples} of fraction {fpfn_analysis_fraction}): {avg(false_negative_percentages):%}')

    spark.stop()



if __name__ == '__main__':

    default_str = ' (default: %(default)s)'

    parser = ArgumentParser()
    parser.add_argument("file", type=str, help="path to the input dataset")
    parser.add_argument("-k", type=int, default=9, help="shingle size" + default_str)
    parser.add_argument("-b", "--bands", type=int, default=13, help="number of LSH bands" + default_str)
    parser.add_argument("-r", "--rows", type=int, default=11, help="number of LSH rows" + default_str)
    parser.add_argument("-s", "--seed", type=int, default=1, help="seed for the random number generator" + default_str)
    parser.add_argument("-p", "--partitions", type=int, default=8, help="number of partitions (mainly for filtering operations)" + default_str)
    parser.add_argument("--similar-to", type=str, default=None, help="if set, return the most similar tweets to the provided tweet" + default_str)
    parser.add_argument("--fpfn-analysis", action="store_true", help="perform analysis on FP (false positives) and FN (false negatives) on samples of the dataset")
    parser.add_argument("--fpfn-analysis-samples", type=int, default=1, help="number of samples for the FP/FN analysis" + default_str)
    parser.add_argument("--fpfn-analysis-fraction", type=float, default=0.1, help="fraction of the dataset to use for each sample for the FP/FN analysis" + default_str)
    parser.add_argument("--minhash-base", type=str, default="minhash", help="base name for the minhash calculations" + default_str)
    parser.add_argument("--candidate-pairs-base", type=str, default="candidate_pairs", help="base name for the candidate pairs (without false positives)" + default_str)
    
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
        fpfn_analysis_fraction=args.fpfn_analysis_fraction,
        minhash_base=args.minhash_base,
        candidate_pairs_base=args.candidate_pairs_base,
    )
