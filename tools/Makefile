.PHONY: all clean

all: venv37pt131cu10/bin/activate venv37pt131cu10/lib/python3.7/site-packages/torch
#all: venv36pt131cu10/bin/activate venv36pt131cu10/lib/python3.6/site-packages/torch

venv37pt131cu10/bin/activate:
	test -d venv37pt131cu10 || virtualenv -p python3.7 venv37pt131cu10
	. venv37pt131cu10/bin/activate; pip install -r requirements.txt; pip install matplotlib;
	touch venv37pt131cu10/bin/activate

#venv36pt131cu10/bin/activate:
#	test -d venv36pt131cu10 || virtualenv -p python3.6 venv36pt131cu10
#	. venv36pt131cu10/bin/activate; pip install -r requirements.txt; pip install matplotlib;
#	touch venv36pt131cu10/bin/activate

venv37pt131cu10/lib/python3.7/site-packages/torch: venv37pt131cu10/bin/activate
	#### For cuda10.1
	#. venv37pt131cu10/bin/activate; pip install pip --upgrade; pip3 install torch torchvision
	#### For cuda10.0
	### For python3.7
	. venv37pt131cu10/bin/activate; pip install pip --upgrade; \
        pip3 install https://download.pytorch.org/whl/cu100/torch-1.3.1%2Bcu100-cp37-cp37m-linux_x86_64.whl; \
            pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.4.2%2Bcu100-cp37-cp37m-linux_x86_64.whl
	### see https://pytorch.org/ or
	### https://download.pytorch.org/whl/cu100/torch_stable.html or
	### https://download.pytorch.org/whl/cu92/torch_stable.html

#venv36pt131cu10/lib/python3.6/site-packages/torch: venv36pt131cu10/bin/activate
#	#### For cuda10.1
#	#. venv36pt131cu10/bin/activate; pip install pip --upgrade; pip3 install torch torchvision
#	#### For cuda10.0
#	### For python3.6
#	. venv36pt131cu10/bin/activate; pip install pip --upgrade; \
#        pip3 install https://download.pytorch.org/whl/cu100/torch-1.3.1%2Bcu100-cp36-cp36m-linux_x86_64.whl; \
#            pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.4.2%2Bcu100-cp36-cp36m-linux_x86_64.whl
#	### see https://pytorch.org/ or
#	### https://download.pytorch.org/whl/cu100/torch_stable.html or
#	### https://download.pytorch.org/whl/cu92/torch_stable.html

clean:
	rm -fr venv37pt131cu10
	#rm -fr venv36pt131cu10
	find -iname "*.pyc" -delete
