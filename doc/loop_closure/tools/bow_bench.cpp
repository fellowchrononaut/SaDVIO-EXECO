// Bag-of-words loop detection benchmark (phase 0): keyframe-to-keyframe DBoW3 similarity matrices.
//
// Built in the SaDVIO container against DBoW3 (BSD) and OpenCV; reads the keyframe lists of make_kf_sets.py (images
// only, no ground truth). For each feature setting and sequence it writes <out>/<setting>/<seq>_sim.npy (float32, L1
// BoW score of every keyframe pair) and <out>/<setting>/<seq>_time.txt (ms per image to detect, describe and
// transform; ms per database query against all the sequence's keyframes).
//
// Settings:
//   fast150   FAST (threshold 10, non-max) on a grid, best per cell, ~150 points, ORB descriptor at full resolution:
//             what the SaDVIO front end already computes (cvFASTFeatureDetector, 150 features, 1 per cell)
//   fast500   the same with ~500 points: the extra keyframe corners of VINS-Mono's loop closure
//   orb1000   ORB with its 8-level pyramid and 1000 points: the ORB-SLAM features the vocabulary was trained on
//
// usage: bow_bench <vocabulary (.txt from ORB-SLAM or .dbow3)> <list dir> <out dir> [seq ...]

#include <DBoW3/DBoW3.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;
using Clock  = std::chrono::steady_clock;

// DBoW3 keeps its ORB-SLAM text loader protected
struct Vocabulary : public DBoW3::Vocabulary {
    using DBoW3::Vocabulary::load_fromtxt;
};

static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static void writeNpy(const fs::path &path, const std::vector<float> &data, size_t n) {
    std::ostringstream h;
    h << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << n << ", " << n << "), }";
    std::string header = h.str();
    while ((10 + header.size() + 1) % 64 != 0)
        header += ' ';
    header += '\n';
    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY\x01\x00", 8);
    const uint16_t len = static_cast<uint16_t>(header.size());
    f.write(reinterpret_cast<const char *>(&len), 2);
    f.write(header.data(), header.size());
    f.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
}

// FAST on a grid: the strongest corner per cell, cells sized for about n points
static std::vector<cv::KeyPoint> gridFast(const cv::Mat &img, int n) {
    std::vector<cv::KeyPoint> all;
    cv::FAST(img, all, 10, true);
    const double cell = std::sqrt(double(img.cols) * img.rows / n);
    const int gx = std::max(1, int(img.cols / cell)), gy = std::max(1, int(img.rows / cell));
    std::vector<int> best(gx * gy, -1);
    for (int i = 0; i < int(all.size()); i++) {
        const int cx = std::min(gx - 1, int(all[i].pt.x / cell)), cy = std::min(gy - 1, int(all[i].pt.y / cell));
        int &b = best[cy * gx + cx];
        if (b < 0 || all[i].response > all[b].response)
            b = i;
    }
    std::vector<cv::KeyPoint> kps;
    for (int b : best)
        if (b >= 0)
            kps.push_back(all[b]);
    return kps;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "usage: bow_bench <vocabulary> <list dir> <out dir> [seq ...]\n";
        return 1;
    }
    const std::string voc_path = argv[1];
    const fs::path lists(argv[2]), out(argv[3]);

    Vocabulary voc;
    auto t0 = Clock::now();
    if (voc_path.size() > 4 && voc_path.substr(voc_path.size() - 4) == ".txt") {
        voc.load_fromtxt(voc_path);
        const std::string bin = (out / "ORBvoc.dbow3").string();
        fs::create_directories(out);
        voc.save(bin, true);
        std::cout << "vocabulary: text loaded in " << ms(t0, Clock::now()) / 1e3 << " s, saved as " << bin << "\n";
        t0 = Clock::now();
        voc.load(bin);
    } else {
        voc.load(voc_path);
    }
    std::cout << "vocabulary: " << voc.size() << " words, loaded in " << ms(t0, Clock::now()) / 1e3 << " s\n";

    std::vector<std::string> seqs;
    for (int i = 4; i < argc; i++)
        seqs.push_back(argv[i]);
    if (seqs.empty())
        for (auto &e : fs::directory_iterator(lists))
            if (e.path().extension() == ".txt")
                seqs.push_back(e.path().stem().string());
    std::sort(seqs.begin(), seqs.end());

    cv::Ptr<cv::ORB> orb_desc = cv::ORB::create();
    cv::Ptr<cv::ORB> orb1000  = cv::ORB::create(1000, 1.2f, 8);
    for (const std::string setting : {"fast150", "fast500", "orb1000"}) {
        fs::create_directories(out / setting);
        for (const auto &seq : seqs) {
            std::ifstream lf(lists / (seq + ".txt"));
            std::vector<std::string> files;
            for (std::string l; std::getline(lf, l);)
                if (!l.empty())
                    files.push_back(l);

            std::vector<DBoW3::BowVector> bows(files.size());
            double t_feat = 0;
            size_t n_kps  = 0;
            for (size_t i = 0; i < files.size(); i++) {
                cv::Mat img = cv::imread(files[i], cv::IMREAD_GRAYSCALE);
                auto a      = Clock::now();
                std::vector<cv::KeyPoint> kps;
                cv::Mat desc;
                if (std::string(setting) == "orb1000") {
                    orb1000->detectAndCompute(img, cv::noArray(), kps, desc);
                } else {
                    kps = gridFast(img, std::string(setting) == "fast150" ? 150 : 500);
                    orb_desc->compute(img, kps, desc);
                }
                voc.transform(desc, bows[i]);
                t_feat += ms(a, Clock::now());
                n_kps += kps.size();
            }
            const size_t n = files.size();
            std::vector<float> sim(n * n, 0.f);
            for (size_t i = 0; i < n; i++)
                for (size_t j = i; j < n; j++)
                    sim[i * n + j] = sim[j * n + i] = static_cast<float>(voc.score(bows[i], bows[j]));
            writeNpy(out / setting / (seq + "_sim.npy"), sim, n);

            // Query cost against a database holding all of the sequence's keyframes (an upper bound for a run)
            DBoW3::Database db(voc, false, 0);
            for (auto &b : bows)
                db.add(b);
            auto q0       = Clock::now();
            const int nq  = std::min<int>(50, int(n));
            for (int k = 0; k < nq; k++) {
                DBoW3::QueryResults r;
                db.query(bows[(k * 7919) % n], r, 5);
            }
            const double t_query = ms(q0, Clock::now()) / nq;
            std::ofstream tf(out / setting / (seq + "_time.txt"));
            tf << "n " << n << "\nkps_per_image " << double(n_kps) / n << "\nms_feature_transform "
               << t_feat / n << "\nms_query " << t_query << "\n";
            std::cout << setting << " " << seq << ": " << n << " keyframes, " << double(n_kps) / n << " points, "
                      << t_feat / n << " ms per image, " << t_query << " ms per query\n";
        }
    }
    return 0;
}
