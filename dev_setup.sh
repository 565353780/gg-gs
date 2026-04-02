cd ..
git clone github@github.com:565353780/base-gs-trainer.git

conda install -y conda-forge::cgal

cd base-gs-trainer
./dev_setup.sh

cd ../gg-gs/submodules/diff-gaussian-rasterization
python setup.py install

cd ../warp-patch-ncc
python setup.py install

cd ../tetra_triangulation
python setup.py install
