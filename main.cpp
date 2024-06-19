#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <chrono>
#include <deal.II/base/mpi.h>
#include <deal.II/dofs/dof_tools.h>
#include <iostream>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/base/conditional_ostream.h>

using namespace dealii;

class HeatEquation
{
public:
    HeatEquation();
    void run();

private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void solve();
    void output_results() const;
    Triangulation<3> triangulation;
    FE_Q<3> fe;
    DoFHandler<3> dof_handler;

    TrilinosWrappers::SparseMatrix system_matrix;
    TrilinosWrappers::MPI::Vector solution;
    TrilinosWrappers::MPI::Vector system_rhs;
};

void HeatEquation::run()
{
    make_grid();
    setup_system();

    cout << "\tNumber of degrees of freedom: " << dof_handler.n_dofs() << " (by partition:";
    for (unsigned int p = 0; p < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++p)
        cout << (p == 0 ? ' ' : '+') << (DoFTools::count_dofs_with_subdomain_association(dof_handler, p));
    cout << ')' << std::endl;

    assemble_system();
    solve();
    output_results();
}

HeatEquation::HeatEquation()
    : fe(1), triangulation(), dof_handler(triangulation)
{
}

void HeatEquation::make_grid()
{
    const Point<3> bottom_left(0, 0, 0);
    const Point<3> top_right(1, 1, 1);
    const unsigned int n_divisions = 100;

    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              {n_divisions, n_divisions, n_divisions},
                                              bottom_left,
                                              top_right);
}

void HeatEquation::setup_system()
{
    GridTools::partition_triangulation(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), triangulation);
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::subdomain_wise(dof_handler);

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp);

    const std::vector<IndexSet> locally_owned_dofs_per_proc =
        DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
    const IndexSet &locally_owned_dofs =
        locally_owned_dofs_per_proc[Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)];

    std::cout << locally_owned_dofs.size() << std::endl;

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         MPI_COMM_WORLD);

    solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
}

void HeatEquation::assemble_system()
{
    system_matrix = 0;
    system_rhs = 0;

    QGauss<3> quadrature_formula(2);
    FEValues<3> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (cell->subdomain_id() == Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        {
            cell_matrix = 0;
            cell_rhs = 0;
            fe_values.reinit(cell);

            for (unsigned int q_index = 0; q_index < quadrature_formula.size(); ++q_index)
            {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        cell_matrix(i, j) += (fe_values.shape_grad(i, q_index) *
                                              fe_values.shape_grad(j, q_index) *
                                              fe_values.JxW(q_index));
                    }
                }
            }

            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
                }
            }
        }
    }

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ConstantFunction<3>(100.0),
                                             boundary_values);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
                                             Functions::ConstantFunction<3>(0.0),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
}

void HeatEquation::solve()
{

    SolverControl solver_control(1000000, 1e-5);
    TrilinosWrappers::SolverCG solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, TrilinosWrappers::PreconditionIdentity());
}

void HeatEquation::output_results() const
{
    DataOut<3> data_out;
    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(solution, "temperature");

    data_out.build_patches();

    std::ofstream output("temperature.vtk");
    data_out.write_vtk(output);
}

int main(int argc, char *argv[])
{
    try
    {
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

        HeatEquation heat_equation_solver;
        auto start = std::chrono::high_resolution_clock::now();
        heat_equation_solver.run();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "Time taken by calc: " << double(duration.count()) / 1000 << " ms" << std::endl;
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "Exception on processing: " << std::endl
                  << exc.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "Unknown exception!" << std::endl;
        return 1;
    }

    return 0;
}
