data_var = data processed raw
# report_var = report figures
src_var = src dataset modules models visualization runtime 
#have to use all for multiple targets
all: data_var src_var docs 
data_var: $(data_var)
	echo "finished data!!!"
# report_var: $(report_var)
# 	echo "finished report!!!"
src_var: $(src_var)
	echo "finished src!!!"
data: 
	mkdir data
processed:
	mkdir data/processed
raw:
	mkdir data/raw
figures:
	mkdir report/figures
src:
	mkdir src
	touch src/data_loader.py
	touch src/early_stopping.py
	touch src/main.py
	touch src/trainer.py
	touch src/infer.py
	touch src/utils.py
dataset:
	mkdir src/dataset
models:
	mkdir src/models
modules:
	mkdir src/modules
runtime:
	mkdir src/runtime
visualization:
	touch src/visualize.ipynb
docs: 
	mkdir docs
deploy:
	mkdir deploy


	
	
