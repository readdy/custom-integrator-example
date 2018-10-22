/**
 * @file binding.cpp
 * @brief Example of a custom integrator, modeling spatially anisotropic diffusion
 * @author chrisfroe
 * @author clonker
 * @date 22.10.18
 */

#include <pybind11/pybind11.h>

#include <readdy/kernel/singlecpu/SCPUKernel.h>
#include <readdy/model/actions/UserDefinedAction.h>
#include <readdy/common/boundary_condition_operations.h>

namespace py = pybind11;
namespace rnd = readdy::model::rnd;

class SCPUAnisotropicBD : public readdy::model::actions::UserDefinedAction {
public:
    SCPUAnisotropicBD(readdy::scalar timeStep, readdy::scalar alphaX, readdy::scalar alphaY, readdy::scalar alphaZ)
            : UserDefinedAction(timeStep), alphaX(alphaX), alphaY(alphaY), alphaZ(alphaZ) {};

    /**
     * Compared to standard BD, where the diffusion matrix is unity times a constant, here we have a diagonal matrix
     * that independently scale the x, y and z component of diffusion. These scaling factors are alphaX, -Y, and -Z.
     */
    void perform(const readdy::util::PerformanceNode &node) override {
        auto scpuKernel = dynamic_cast<readdy::kernel::scpu::SCPUKernel *>(kernel());

        auto t = node.timeit();
        const auto &context = scpuKernel->context();
        const auto &pbc = context.periodicBoundaryConditions().data();
        const auto &kbt = context.kBT();
        const auto &box = context.boxSize().data();
        auto &stateModel = scpuKernel->getSCPUKernelStateModel();
        const auto pd = stateModel.getParticleData();

        for (auto &entry : *pd) {
            if (!entry.is_deactivated()) {
                const readdy::scalar D = context.particleTypes().diffusionConstantOf(entry.type);
                auto randomDisplacement = std::sqrt(2. * D * _timeStep) *
                                          (readdy::model::rnd::normal3<readdy::scalar>());
                randomDisplacement.x *= std::sqrt(alphaX);
                randomDisplacement.y *= std::sqrt(alphaY);
                randomDisplacement.z *= std::sqrt(alphaZ);
                entry.pos += randomDisplacement;

                auto deterministicDisplacement = entry.force * _timeStep * D / kbt;
                deterministicDisplacement.x *= alphaX;
                deterministicDisplacement.y *= alphaY;
                deterministicDisplacement.z *= alphaZ;
                entry.pos += deterministicDisplacement;

                readdy::bcs::fixPosition(entry.pos, box, pbc);
            }
        }
    }

private:
    const readdy::scalar alphaX, alphaY, alphaZ;
};


PYBIND11_MODULE (myintegrator, m) {
    py::module::import("readdy");
    py::class_<SCPUAnisotropicBD, readdy::model::actions::UserDefinedAction,
            std::shared_ptr<SCPUAnisotropicBD>>(m, "SCPUAnisotropicBD").def(
            py::init<readdy::scalar, readdy::scalar, readdy::scalar, readdy::scalar>());
}
