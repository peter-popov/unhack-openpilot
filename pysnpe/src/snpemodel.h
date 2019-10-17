#ifndef SNPEMODEL_H
#define SNPEMODEL_H

#include <SNPE/SNPE.hpp>
#include <SNPE/SNPEBuilder.hpp>
#include <SNPE/SNPEFactory.hpp>
#include <DlContainer/IDlContainer.hpp>
#include <DlSystem/DlError.hpp>
#include <DlSystem/ITensor.hpp>
#include <DlSystem/ITensorFactory.hpp>
#include <DlSystem/IUserBuffer.hpp>
#include <DlSystem/IUserBufferFactory.hpp>

class SNPEModel {
public:
  SNPEModel(std::string path, float *output, size_t output_size);

  ~SNPEModel() {
    if (model_data) free(model_data);
  }
  void execute(float *net_input_buf);

private:
  uint8_t *model_data = NULL;

  // snpe model stuff
  std::unique_ptr<zdl::SNPE::SNPE> snpe;

  // snpe input stuff
  zdl::DlSystem::UserBufferMap inputMap;
  std::unique_ptr<zdl::DlSystem::IUserBuffer> inputBuffer;

  // snpe output stuff
  zdl::DlSystem::UserBufferMap outputMap;
  std::unique_ptr<zdl::DlSystem::IUserBuffer> outputBuffer;
  float *output;
};

#endif

