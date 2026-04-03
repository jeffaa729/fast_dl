# Compiler
CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -I ./eigen-5.0.0

# Where to put executables and object files
BINDIR   = bin
OBJDIR   = build

MAIN_SRC = main.cpp

SRCS = $(MAIN_SRC) DataLoader.cpp Layers.cpp Layer.cpp LinearLayer.cpp SigmoidLayer.cpp Optimizer.cpp SoftmaxCrossEntropyLoss.cpp Network.cpp

# Map .cpp -> build/xxx.o
OBJS = $(addprefix $(OBJDIR)/, $(SRCS:.cpp=.o))

# Executable name = bin/main (if MAIN_SRC = main.cpp)
TARGET = $(BINDIR)/$(basename $(MAIN_SRC))

# -------- Default target --------
all: $(BINDIR) $(OBJDIR) $(TARGET)

# Create bin/ and build/ if they don't exist
$(BINDIR):
	mkdir -p $(BINDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

# Link main
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@

# Compile rule: .cpp -> build/xxx.o
$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


# -------- Test targets --------

TEST_TARGET = $(BINDIR)/test_layers
TEST_SRCS   = test_layers.cpp Layers.cpp
TEST_OBJS   = $(addprefix $(OBJDIR)/, $(TEST_SRCS:.cpp=.o))

OOP_TARGET  = $(BINDIR)/test_oop
OOP_SRCS    = test_oop.cpp Layers.cpp Layer.cpp LinearLayer.cpp SigmoidLayer.cpp Optimizer.cpp
OOP_OBJS    = $(addprefix $(OBJDIR)/, $(OOP_SRCS:.cpp=.o))

NET_TARGET  = $(BINDIR)/test_network
NET_SRCS = test_network.cpp Layers.cpp Layer.cpp LinearLayer.cpp SigmoidLayer.cpp Optimizer.cpp SoftmaxCrossEntropyLoss.cpp Network.cpp DataLoader.cpp
NET_OBJS    = $(addprefix $(OBJDIR)/, $(NET_SRCS:.cpp=.o))

MNIST_TARGET  = $(BINDIR)/test_train_mnist
MNIST_SRCS = test_train_mnist.cpp Layers.cpp Layer.cpp LinearLayer.cpp SigmoidLayer.cpp Optimizer.cpp SoftmaxCrossEntropyLoss.cpp Network.cpp ParametricReLU.cpp DataLoader.cpp
MNIST_OBJS    = $(addprefix $(OBJDIR)/, $(MNIST_SRCS:.cpp=.o))

test: $(BINDIR) $(OBJDIR) $(TEST_TARGET) $(OOP_TARGET) $(NET_TARGET) $(MNIST_TARGET)

$(TEST_TARGET): $(TEST_OBJS)
	$(CXX) $(TEST_OBJS) -o $@

$(OOP_TARGET): $(OOP_OBJS)
	$(CXX) $(OOP_OBJS) -o $@

$(NET_TARGET): $(NET_OBJS)
	$(CXX) $(NET_OBJS) -o $@

$(MNIST_TARGET): $(MNIST_OBJS)
	$(CXX) $(MNIST_OBJS) -o $@


# -------- Clean --------

clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean test
