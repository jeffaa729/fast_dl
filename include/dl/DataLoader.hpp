#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>

class DataLoader {
public:
    DataLoader(std::string image_file, std::string label_file, int num);
    DataLoader(std::string image_file, std::string label_file);
    ~DataLoader();

    int getSize() { return size; }
    int getRows() { return rows; }
    int getCols() { return cols; }

    const Eigen::MatrixXf& getImagesMatrix() const { return images; }

    // One image as a row vector: (1 x (rows*cols))
    Eigen::RowVectorXf getImageRow(int idx) const { return images.row(idx); }
    int getLabels(int id) { return labels[id]; }

private:
    Eigen::MatrixXf images;
    std::vector<int> labels;
    int size;
    int rows;
    int cols;

    void load_images(std::string file, int num=0);
    void load_labels(std::string file, int num=0);
    int  to_int(char* p);
};
