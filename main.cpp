#include <array>
#include <iostream>
#include <cstdint>
#include <math.h>
#include <algorithm>
#include <vector>
#include <iomanip>

constexpr std::size_t kFeatureCount = 4;
const std::array<std::string, kFeatureCount> feature_names = {
    "relevance_score",
    "risk_score",
    "freshness_score",
    "confidence_prior"
};

constexpr double kOneBillionParameters = 1000000000.0;

constexpr float kErrorTolerance = 0.01f;

struct QuantizationConfig {
    float real_min;
    float real_max;

    int quantized_min;
    int quantized_max;

    float scale;

    int zero_point;

    float tolerance;
};


struct QuantizedValueReport {
    std::string feature_name;

    float fp32_value;
    float clipped_value;

    std::int8_t int8_payload;

    float dequantized_value;
    float absolute_error;

    bool clipped;
};

struct Int4PackedPair {
    std::string high_feature_name;
    std::string low_feature_name;

    std::uint8_t high_code;
    std::uint8_t low_code;
    std::uint8_t packed_byte;

    float high_dequantized;
    float low_dequantized;
};

QuantizationConfig buildAffineInt8Config(float real_min,
                                         float real_max,
                                         float tolerance) {
    const int qmin = static_cast<int>(std::numeric_limits<std::int8_t>::min());
    const int qmax = static_cast<int>(std::numeric_limits<std::int8_t>::max());

    const float scale = (real_max - real_min) / static_cast<float>(qmax-qmin);

    const int zero_point =
        static_cast<int>(std::round(static_cast<float>(qmin) - (real_min / scale)));

    return {real_min, real_max, qmin, qmax, scale, zero_point, tolerance};
}

QuantizationConfig buildUsignedInt4Config(float real_min,
                                          float real_max,
                                          float tolerance) {
    const int qmin = 0;
    const int qmax = 15;

    const float scale = (real_max - real_min) / static_cast<float>(qmax - qmin);
    const int zero_point = static_cast<int>(std::round(static_cast<float>(qmin) - (real_min / scale)));

    return {real_min, real_max, qmin, qmax, scale, zero_point, tolerance};
}

float clipToRealRange(float value, const QuantizationConfig &config) {
    return std::clamp(value, config.real_min, config.real_max);
}

std::int8_t quantizeToInt8(float value, const QuantizationConfig &config) {
    const float clipped = clipToRealRange(value, config);

    const float raw_quantized = (clipped / config.scale) + static_cast<float>(config.zero_point);

    const int rounded = static_cast<int>(std::round(raw_quantized));

    const int clamped = 
        std::clamp(rounded, config.quantized_min, config.quantized_max);

    return static_cast<std::int8_t>(clamped);
}

float dequantizeFromInt8(std::int8_t payload, const QuantizationConfig &config) {
    const int code = static_cast<int>(payload);
    
    return (static_cast<float>(code - config.zero_point)) * config.scale;
}

std::uint8_t quantizeToUInt4Code(float value,
                                 const QuantizationConfig &config) {
    const float clipped = clipToRealRange(value, config);
    const float raw_quantized = (clipped / config.scale) + static_cast<float>(config.zero_point);

    const int rounded = static_cast<int>(std::round(raw_quantized));
    const int clamped = std::clamp(rounded, config.quantized_min, config.quantized_max);

    return static_cast<std::uint8_t>(clamped);
}

std::uint8_t packTwoUInt4(std::uint8_t high_code, std::uint8_t low_code) {
    const std::uint8_t high_nibble = static_cast<std::uint8_t>((high_code & 0x0F) << 4);
    const std::uint8_t low_nibble = static_cast<std::uint8_t>(low_code & 0x0F);

    return static_cast<std::uint8_t>(high_nibble | low_nibble);
}

std::uint8_t unpackHighUInt4(std::uint8_t packed_byte) {
    return static_cast<std::uint8_t>((packed_byte >> 4) & 0x0F);
}

std::uint8_t unpackLowUInt4(std::uint8_t packed_byte) {
    return static_cast<std::uint8_t>(packed_byte & 0x0F);
}

float dequantizeFromUInt4Code(std::uint8_t code, const QuantizationConfig &config) {
    const int promoted_code = static_cast<int>(code);

    return (static_cast<float>(promoted_code - config.zero_point)) * config.scale;
}

QuantizedValueReport buildQuantizedValueReport(const std::string &feature_name,
                                               float fp32_value,
                                               const QuantizationConfig &config) {
    const float clipped_value = clipToRealRange(fp32_value, config);
    const std::int8_t payload = quantizeToInt8(fp32_value, config);

    const float dequantized_value = dequantizeFromInt8(payload, config);
    const float absolute_error = std::fabs(dequantized_value - clipped_value);

    const bool clipped = std::fabs(clipped_value - fp32_value) > 0.0f;

    return {
        feature_name,
        fp32_value,
        clipped_value,
        payload,
        dequantized_value,
        absolute_error,
        clipped
    };

}

void printQuantizationRow(const QuantizedValueReport &report) {
    std::cout << std::left << std::setw(22) << report.feature_name;

    std::cout << " fp32=" << std::fixed << std::setprecision(6) << std::setw(10) << report.fp32_value;
    std::cout << " clipped=" << std::setw(10) << report.clipped_value;
    std::cout << " INT8 payload=" << std::setw(5) << static_cast<int>(report.int8_payload);
    std::cout << " dequantized=" << std::setw(10) << report.dequantized_value;
    std::cout << " abs_error=" << std::setw(10) << report.absolute_error;
    std::cout << " clipped_flags=" << (report.clipped ? "yes" : "no") << std::endl;

}

bool evaluateQualityGate(const std::vector<QuantizedValueReport> &reports,
                         float tolerance,
                         float &max_error,
                         float &average_error) {
    max_error = 0.0f;

    float total_error = 0.0f;

    for (const QuantizedValueReport &report : reports) {
        max_error = std::max(max_error, report.absolute_error);

        total_error += report.absolute_error;
    }

    average_error = reports.empty() ? 0.0f : total_error / static_cast<float>(reports.size());

    return max_error <= tolerance;
}

int main()
{
    std::cout << "Part 1\n" << std::endl;
    return 0;
}