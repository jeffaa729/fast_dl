#include "DataLoader.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>

DataLoader::DataLoader(std::string image_file,
			   std::string label_file,
			   int num) : size(0), rows(0), cols(0) 
{
    load_images(image_file, num);
    load_labels(label_file, num);
}

DataLoader::DataLoader(std::string image_file,
			            std::string label_file) :
DataLoader(image_file, label_file, 0) {}

DataLoader::~DataLoader() {}

int DataLoader::to_int(char* p) {
    return ((p[0] & 0xff) << 24) | ((p[1] & 0xff) << 16) |
            ((p[2] & 0xff) <<  8) | ((p[3] & 0xff) <<  0);
}

void DataLoader::load_images(std::string image_file, int num) {
    std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Could not open image file: " + image_file);
    }

    char p[4];

    // Magic number
    ifs.read(p, 4);
    int magic_number = to_int(p);
    assert(magic_number == 0x803);  

    // Number of images
    ifs.read(p, 4);
    size = to_int(p);
    if (num != 0 && num < size) size = num;  

    // Rows
    ifs.read(p, 4);
    rows = to_int(p);

    // Columns
    ifs.read(p, 4);
    cols = to_int(p);

    int image_size = rows * cols;

    // Allocate Eigen matrix: (size x image_size)
    images = Eigen::MatrixXf(size, image_size);

    // Temporary buffer for one image (raw bytes)
    char* q = new char[image_size];

    for (int i = 0; i < size; ++i) {
        ifs.read(q, image_size);

        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel = static_cast<unsigned char>(q[j]);
            images(i, j) = static_cast<float>(pixel) / 255.0f;  
        }
    }

    delete[] q;
    ifs.close();
}

void DataLoader::load_labels(std::string label_file, int num) {
    std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);
    char p[4];

    ifs.read(p, 4);
    int magic_number = to_int(p);
    assert(magic_number == 0x801);

    ifs.read(p, 4);
    int size = to_int(p);
    // limit
    if (num != 0 && num < size) size = num;

    for (int i=0; i<size; ++i) {
        ifs.read(p, 1);
        int label = p[0];
        labels.push_back(label);
    }

    ifs.close();
}