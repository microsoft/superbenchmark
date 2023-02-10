CC= /opt/AMD/aocc-compiler-4.0.0/bin/clang
CFLAGS= -Ofast -mcmodel=large -mavx2 -ffp-contract=fast -lomp -fopenmp -fnt-store=aggressive -DNTIMES=10
ZEN3FLAGS= -march=znver3 
ZEN3_OUTPUT= xhpl_z3
ZEN4_OUTPUT= xhpl_z4 

all: ZEN3 ZEN4

ZEN3: stream.c
	$(CC) $(CFLAGS) $(ZEN3FLAGS) stream.c -o $(ZEN3_OUTPUT)
ZEN4:
	wget https://download.amd.com/developer/eula/pre-built/hpl/amd-zen-hpl-2022_11.tar.gz && \
	tar xzf amd-zen-hpl-2022_11.tar.gz 
	mv amd-zen-hpl-2022_11/xhpl ./$(ZEN4_OUTPUT)
clean:
	rm $(GEN_OUTPUT) $(ZEN3_OUTPUT) $(ZEN4_OUTPUT)