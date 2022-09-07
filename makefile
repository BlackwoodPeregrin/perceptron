CC = g++ -std=c++17
CFLAGS = -Wall -Wextra -Werror
LFLAGS = -lgtest
BD=build
INSTALL=install
BACK=backup_perceptron

OS = $(shell uname -s)
ifeq ($(OS), Linux)
	LFLAGS += -pthread -lsubunit
	OPEN = -xdg-open
else
	OPEN = -open
endif

PROFILE = perceptron.pro

all : clean build

build : mk replication
	cd $(BD); qmake $(PROFILE)
	make -C $(BD)/
	cd $(BD); rm -rf *.c *.h *.hpp *.cpp *.ui *.o *.icns *.pro *.user *.qrc Makefile icons_iu .qmake.stash

mk :
	-mkdir $(BD)/
	-mkdir $(BD)/icons_iu

replication :
ifeq ($(OS), Linux)
	cp -r icons_iu/ $(BD)
else
	-mkdir $(BD)/icons_iu
	cp -r icons_iu/ $(BD)/icons_iu
	-mkdir $(BD)/train_test_files
	cp -r train_test_files/ $(BD)/train_test_files
	-mkdir $(BD)/weights_files
	cp -r weights_files/ $(BD)/weights_files
endif
	cp $(PROFILE) $(BD)
	cp $(PROFILE).user $(BD)
	cp *.cpp $(BD)
	cp *.h $(BD)
	cp *hpp $(BD)
	cp *.ui $(BD)
	cp *.qrc $(BD)	

install : uninstall build
	-mkdir $(INSTALL) $(INSTALL)/train_test_files $(INSTALL)/weights_files
	cp -r $(BD)/ $(INSTALL)/
	cp -r train_test_files/ $(INSTALL)/train_test_files
	cp -r weights_files/ $(INSTALL)/weights_files

uninstall :
	rm -rf $(INSTALL)

dist : clean
	rm -rf $(BACK) 
	mkdir $(BACK)
	mkdir $(BACK)/src
	cp ../*.md $(BACK)
ifeq ($(OS), Linux)
	cp -r ../misc $(BACK)
	cp -r ../materials $(BACK)
	cp -r icons_iu $(BACK)/src
	cp -r train_test_files $(BACK)/src
	cp -r weights_files $(BACK)/src

else
	-mkdir $(BACK)/misc
	cp -r ../misc/ $(BACK)/misc
	-mkdir $(BACK)/materials
	cp -r ../materials/ $(BACK)/materials
	-mkdir $(BACK)/src/icons_iu
	cp -r icons_iu/ $(BACK)/src/icons_iu
	-mkdir $(BACK)/src/train_test_files
	cp -r train_test_files/ $(BACK)/src/train_test_files
	-mkdir $(BACK)/src/weights_files
	cp -r weights_files/ $(BACK)/src/weights_files
endif
	cp *.cpp $(BACK)/src
	cp *.h $(BACK)/src
	cp *hpp $(BACK)/src
	cp *.ui $(BACK)/src
	cp $(PROFILE) $(BACK)/src
	cp *.user $(BACK)/src
	cp makefile $(BACK)/src
	cp *.qrc $(BACK)/src
	tar -cvzf $(HOME)/Desktop/dist_perceptron.tgz $(BACK)/
	rm -rf $(BACK)

test : tests

tests : clean
	$(CC) $(FLAGS) tests.cpp neuron.cpp -o test $(LFLAGS)
	./test

leaks :	tests
	leaks --atExit -- ./test

cpplint :
	cp ../materials/linters/CPPLINT.cfg ./
	-python3 ../materials/linters/cpplint.py --extensions=cpp *.cpp
	-python3 ../materials/linters/cpplint.py --extensions=hpp *.hpp
	-python3 ../materials/linters/cpplint.py --extensions=h *.h
	$(RM) CPPLINT.cfg

cppcheck :
	cppcheck --std=c++17 --enable=all --check-config --suppress=missingIncludeSystem --suppress=missingInclude --suppress=unmatchedSuppression *.cpp *.hpp *.h

gcov : gcov_report

gcov_report : clean
	$(MAKE) LFLAGS="$(LFLAGS) --coverage" tests
	lcov -t test -o test.info -c -d . --no-external
	genhtml -o report test.info
	$(OPEN) report/index.html

report_clean :
	$(RM) -rf ./*.gcda ./*.gcno ./*.info ./*.gch ./report

dvi :
	-makeinfo --html --force man.tex
	$(OPEN) man/index.html

clean : report_clean
	$(RM) -rf test *.a *.so *.o *.cfg *.gcda *.gcno *.html *.info *.dSYM report man
	$(RM) -rf $(BD)
