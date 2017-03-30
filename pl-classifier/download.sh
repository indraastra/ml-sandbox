mkdir -p tmp
mkdir -p data
cd tmp/
git clone --depth 1 https://github.com/nodejs/node.git
git clone --depth 1 https://github.com/tensorflow/tensorflow.git
git clone --depth 1 https://github.com/rails/rails.git
git clone --depth 1 https://github.com/opencv/opencv.git
git clone --depth 1 https://github.com/apache/spark.git
git clone --depth 1 https://github.com/google/guava.git
git clone --depth 1 https://github.com/torch/torch7
git clone --depth 1 https://github.com/jgm/lunamark
cd ../

find_ext() { find ../tmp/ -name *.$1; }
cp_files() { xargs -I {} cp {} $1; }

cd data/
for lang in py cpp java ruby js css html c lua
do
	mkdir $lang
	find_ext $lang | cp_files $lang
done
cd ../
