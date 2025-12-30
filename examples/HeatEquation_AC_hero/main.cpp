/*
 * A simplified single file version of the HeatEquation_EX0_C exmaple.
 * This code is designed to be used with Demo_Tutorial.rst.
 *
 */

#include <AMReX.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include "thermal_properties.h"

bool retrain_surrogate()
{
  PyObject *surrogate = PyObject_GetAttrString(ac_driver, "surrogate");
  if (surrogate == NULL) {
    amrex::Print() << "get surrogate failed" << std::endl;
    PyErr_Print();
  }
  PyObject *dataset = PyObject_GetAttrString(ac_driver, "dataset");
  if (dataset == NULL) {
    amrex::Print() << "get dataset failed" << std::endl;
    PyErr_Print();
  }
  PyObject *x_data = PyObject_GetAttrString(dataset, "x_data");
  if (x_data == NULL) {
    amrex::Print() << "get x_data failed" << std::endl;
    PyErr_Print();
  }
  PyObject *y_data = PyObject_GetAttrString(dataset, "y_data");
  if (y_data == NULL) {
    amrex::Print() << "get y_data failed" << std::endl;
    PyErr_Print();
  }
  PyObject *ret = PyObject_CallMethod(surrogate, "train", "O,O", x_data, y_data);
  if (ret == NULL) {
    amrex::Print() << "train failed" << std::endl;
    PyErr_Print();
    return false;
  }

#if 0
  for (int p=0; p<amrex::ParallelDescriptor::NProcs(); ++p) {
    if (p == amrex::ParallelDescriptor::MyProc())
    {
      std::cout << "PROC " << p << std::endl;
      PyObject *tr = PyObject_CallMethod(ac_driver, "print_dataset", NULL);
    }
    amrex::ParallelDescriptor::Barrier();
  }
#endif
  Py_DECREF(ret);
  Py_DECREF(y_data);
  Py_DECREF(x_data);
  Py_DECREF(dataset);
  Py_DECREF(surrogate);

  return true;
}

void pretrain_kappa_model(const amrex::MultiFab& phi, amrex::Real tol)
{
  // Iteratively find "most informative" point over MultiFab (phi that gives the largest variance, provided that variance is larger
  // that the input tolerance (tol)).  If that point is unique from the surrogate model training set, add the point and retrain.
  // Continue until either no points are found to violate the tolerance, or any points that do are not unique to the training set.
  bool do_iteration = true;
  while (do_iteration)
  {
    amrex::ValLocPair<amrex::Real,amrex::Real> var; // value=max_variance, index=phi at max_variance
    var.value = -1; // Assume valid variance is never < 0.  If we end up < 0, no variances will trigger retraining on this proc

    for ( amrex::MFIter mfi(phi); mfi.isValid(); ++mfi )
    {
      const auto& phi_arr = phi.array(mfi);
      auto bxg=amrex::grow(mfi.validbox(),1);

      // Wrap fab data in a PyTuple (no-copy)
      npy_intp dims[2] = {bxg.numPts(),1}; // Each point will be a 1-long vector of temperature
      PyObject* x_queries = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, const_cast<amrex::Real*>(phi_arr.dataPtr()));
      if (!x_queries) {
	amrex::Abort("x_queries create failed");
      }

      // Call ac_driver.query_for_invalid, returns a PyLong pointing to location of highest variance, if above threshold
      PyObject *rTuple = PyObject_CallMethod(ac_driver, "query_for_invalid", "O,s,d", x_queries, "absolute_variance", tol);
      if (rTuple == NULL) {
	amrex::Abort("query_for_invalid failed");
      } else {
	if (!(PyTuple_Check(rTuple) && PyTuple_Size(rTuple) == 2)) {
	  amrex::Abort("query_for_invalid did not return a tuple of length 2");
	}
	long idx_invalid     = PyLong_AsLong(   PyTuple_GetItem(rTuple, 0));
	amrex::Real variance = PyFloat_AsDouble(PyTuple_GetItem(rTuple, 1));

	if (idx_invalid >= 0)
	{
	  var.value = std::max(var.value, variance);
	  var.index = get_double_from_entry(x_queries,idx_invalid,0);
	}
      }
      Py_DECREF(x_queries);
      Py_DECREF(rTuple);
    }

    amrex::ParallelAllReduce::Max(var,amrex::ParallelDescriptor::Communicator());

    // If a point is identified, ensure that it is unique from the training set, then add and retrain the surrogate
    if (var.value > 0)
    {
      npy_intp dims1pt[2] = {1,1};
      PyObject* x_query = PyArray_SimpleNew(2, dims1pt, NPY_DOUBLE);
      if (x_query == NULL) {
	amrex::Abort("x_query(pt)  create failed");
      }
      set_double_at_entry(x_query,0,0,var.index);

      PyObject *uniquify_pts = PyObject_CallMethod(ac_driver, "remove_existing_points", "O,i", x_query, 0);
      if (uniquify_pts == NULL) {
	amrex::Abort("remove_existing_points failed");
      } else {
	if (PyObject_Length(uniquify_pts) > 0) {
	  PyObject *add_pts = PyObject_CallMethod(ac_driver, "add_points", "O,i", x_query, 0);
	  if (add_pts == NULL) {
	    amrex::Abort("add_points failed");
	  } else {
	    amrex::Print() << "RETRAINING SURROGATE..." << std::endl;
	    retrain_surrogate();
	  }
	  Py_DECREF(add_pts);
	} else {
	  do_iteration = false; // i.e., the point found to violate the tol is already in the training set (likely that tol is smaller than achieveable in model due to roundoff/truncation)
	}
      }
      Py_DECREF(x_query);
      Py_DECREF(uniquify_pts);
    } else {
      do_iteration = false; // i.e., no points were found to violate the tolerance
    }
  }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {

    Py_Initialize();
    // Import sys module
    PyObject *sys = PyImport_ImportModule("sys");
    if (!sys) { PyErr_Print(); return -1; }

    // Append the working directory to sys.path
    PyObject *path = PyObject_GetAttrString(sys, "path");
    PyObject *cwd = PyUnicode_FromString(".");
    PyList_Append(path, cwd);
    Py_DECREF(cwd);
    Py_DECREF(path);
    Py_DECREF(sys);

    // Initialize infrastructure for managing numpy arrays
    import_array();

    // Import py_query module
    PyObject *py_thermal_properties = PyImport_ImportModule("py_thermal_properties");
    if (!py_thermal_properties) { PyErr_Print(); return -1; }

    // Call initialize_driver()
    ac_driver = PyObject_CallMethod(py_thermal_properties, "initialize_driver", nullptr);
    if (!ac_driver) { PyErr_Print(); Py_DECREF(py_thermal_properties); return -1; }

    // Call print_data(ac_driver)
    //PyObject *temp = PyObject_CallMethod(py_thermal_properties, "print_data", "O", ac_driver);
    //if (!temp) { PyErr_Print(); Py_DECREF(ac_driver); Py_DECREF(py_thermal_properties); return -1; }
    //Py_DECREF(temp);

    // **********************************
    // DECLARE SIMULATION PARAMETERS
    // **********************************

    // number of cells on each side of the domain
    int n_cell;

    // size of each box (or grid)
    int max_grid_size;

    // total steps in simulation
    int nsteps;

    // how often to write a plotfile
    int plot_int;

    // time step
    amrex::Real dt;

    // surrogate tolerance
    amrex::Real surrogate_tolerance = 1.e-9;

    // **********************************
    // READ PARAMETER VALUES FROM INPUT DATA
    // **********************************
    // inputs parameters
    {
        // ParmParse is way of reading inputs from the inputs file
        // pp.get means we require the inputs file to have it
        // pp.query means we optionally need the inputs file to have it - but we must supply a default here
        amrex::ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of
        //   a square (or cubic) domain.
        pp.get("n_cell",n_cell);

        // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size",max_grid_size);

        // Default nsteps to 10, allow us to set it to something else in the inputs file
        nsteps = 10;
        pp.query("nsteps",nsteps);

        // Default plot_int to -1, allow us to set it to something else in the inputs file
        //  If plot_int < 0 then no plot files will be written
        plot_int = -1;
        pp.query("plot_int",plot_int);

        // time step
        pp.get("dt",dt);

    }

    // **********************************
    // DEFINE SIMULATION SETUP AND GEOMETRY
    // **********************************

    // make BoxArray and Geometry
    // ba will contain a list of boxes that cover the domain
    // geom contains information such as the physical domain size,
    // number of points in the domain, and periodicity
    amrex::BoxArray ba;
    amrex::Geometry geom;

    // define lower and upper indices
    amrex::IntVect dom_lo(AMREX_D_DECL(0,0,0));
    amrex::IntVect dom_hi(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1));

    // Make a single box that is the entire domain
    amrex::Box domain(dom_lo, dom_hi);

    // Initialize the boxarray "ba" from the single box "domain"
    ba.define(domain);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.maxSize(max_grid_size);

    // This defines the physical box, [0,1] in each direction.
    amrex::RealBox real_box({ AMREX_D_DECL(0., 0., 0.)},
			    { AMREX_D_DECL(1., 1., 1.)});

    // periodic in all direction
    amrex::Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};

    // This defines a Geometry object
    geom.define(domain, real_box, amrex::CoordSys::cartesian, is_periodic);

    // extract dx from the geometry object
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // Ncomp = number of components for each array
    int Ncomp = 1;

    // How Boxes are distrubuted among MPI processes
    amrex::DistributionMapping dm(ba);

    // we allocate two phi multifabs; one will store the old state, the other the new.
    amrex::MultiFab phi_old(ba, dm, Ncomp, Nghost);
    amrex::MultiFab phi_new(ba, dm, Ncomp, Nghost);

    // define a T dependent thermal diffusivity
    amrex::MultiFab kappa(ba, dm, Ncomp, Nghost);
    amrex::MultiFab Cp(ba, dm, Ncomp, 0);
    amrex::MultiFab rho(ba, dm, Ncomp, 0);

    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> kappa_face;
    for (int i=0; i<AMREX_SPACEDIM; ++i){
      amrex::BoxArray baf= amrex::BoxArray(ba).surroundingNodes(i);
      kappa_face[i].define(baf,dm,Ncomp,0);
    }

    // time = starting time in the simulation
    amrex::Real time = 0.0;

    // **********************************
    // INITIALIZE DATA LOOP
    // **********************************

    // loop over boxes
    for (amrex::MFIter mfi(phi_old); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();

        const amrex::Array4<amrex::Real>& phiOld = phi_old.array(mfi);

        // set phi = 1 + e^(-(r-0.5)^2)
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            // **********************************
            // SET VALUES FOR EACH CELL
            // **********************************
	  amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> x = {AMREX_D_DECL((i+0.5) * dx[0],
									(j+0.5) * dx[1],
									(k+0.5) * dx[2])};
	  amrex::Real rsd = AMREX_D_TERM(+(x[0]-0.5)*(x[0]-0.5),
					 +(x[1]-0.5)*(x[1]-0.5),
					 +(x[2]-0.5)*(x[2]-0.5));
	  phiOld(i,j,k) = 300. + 500. * std::exp(-rsd/.01);
        });
    }

    // **********************************
    // WRITE INITIAL PLOT FILE
    // **********************************

    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int step = 0;
        const std::string& pltfile = amrex::Concatenate("plt",step,5);
        WriteSingleLevelPlotfile(pltfile, phi_old, {"phi"}, geom, time, 0);
    }


    // **********************************
    // MAIN TIME EVOLUTION LOOP
    // **********************************
    for (int step = 1; step <= nsteps; ++step)
    {
        // fill periodic ghost cells
        phi_old.FillBoundary(geom.periodicity());

	// "pre-train" the model to ensure that all evaluations over the MultiFab can be done without violating the error threshold
	pretrain_kappa_model(phi_old, surrogate_tolerance);

        // new_phi = old_phi + dt * Laplacian(old_phi)
        // loop over boxes

	// Get kappa at cell centers (incl grow) using pre-trained model
	kappa.setVal(-100,0,1);
	for ( amrex::MFIter mfi(phi_old); mfi.isValid(); ++mfi )
        {
            const auto& phi_arr = phi_old.array(mfi);
            const auto& kappa_arr = kappa.array(mfi);
            auto bxg=amrex::grow(mfi.validbox(),1);

	    npy_intp dims[2] = {bxg.numPts(),1}; // Each point will be a 1-long vector of temperature
	    PyObject* x_queries = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, const_cast<amrex::Real*>(phi_arr.dataPtr()));
	    if (!x_queries) {
	      amrex::Abort("x_queries create failed");
	    }
	    PyObject* y_queries = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, const_cast<amrex::Real*>(kappa_arr.dataPtr()));
	    if (!y_queries) {
	      amrex::Abort("y_queries create failed");
	    }

	    // Fill values directly into kappa
	    PyObject *ret = PyObject_CallMethod(ac_driver, "query_assuming_valid", "O,O", x_queries, y_queries);
	    if (ret == NULL) {
	      amrex::Abort("query_assuming_valid failed");
	    }
	    Py_DECREF(x_queries);
	    Py_DECREF(y_queries);
	    Py_DECREF(ret);
	};
	//amrex::Print()<<"min conductivity="<<kappa.min(0)<<std::endl;
	//amrex::Print()<<"max conductivity="<<kappa.max(0)<<std::endl;

	// Get Cp and rho over valid region
	for ( amrex::MFIter mfi(phi_old); mfi.isValid(); ++mfi )
        {
            const amrex::Box& bx = mfi.validbox();
            const amrex::Array4<amrex::Real>& phiOld = phi_old.array(mfi);
            const amrex::Array4<amrex::Real>& Cp_i = Cp.array(mfi);
            const amrex::Array4<amrex::Real>& rho_i = rho.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Cp_i(i,j,k)    = get_SpecificHeatCapacity(phiOld(i,j,k));
                rho_i(i,j,k)   = get_density(phiOld(i,j,k));
            });
	};

	const bool use_harmonic_averaging = true;
	amrex::average_cellcenter_to_face(GetArrOfPtrs(kappa_face),kappa,geom,1,use_harmonic_averaging);

        for ( amrex::MFIter mfi(phi_old); mfi.isValid(); ++mfi )
        {
            // advance the data by dt
            const amrex::Box& bx = mfi.validbox();
            const auto& phiOld = phi_old.array(mfi);
            const auto& phiNew = phi_new.array(mfi);
	    const auto& Cp_i = Cp.array(mfi);
            const auto& rho_i = rho.array(mfi);
	    amrex::GpuArray<const amrex::Array4<amrex::Real>,AMREX_SPACEDIM> kappa_fc
	      = {AMREX_D_DECL(kappa_face[0].array(mfi),
			      kappa_face[1].array(mfi),
			      kappa_face[2].array(mfi))};
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // **********************************
                // EVOLVE VALUES FOR EACH CELL
                // **********************************

                phiNew(i,j,k) = phiOld(i,j,k) + dt / (rho_i(i,j,k)*Cp_i(i,j,k)) *
		    ( ( kappa_fc[0](i+1,j,k)*(phiOld(i+1,j,k) - phiOld(i  ,j,k))
		    -   kappa_fc[0](i  ,j,k)*(phiOld(i  ,j,k) - phiOld(i-1,j,k)) )/(dx[0]*dx[0])
#if AMREX_SPACEDIM > 1
                    + ( kappa_fc[1](i,j+1,k)*(phiOld(i,j+1,k) - phiOld(i,j  ,k))
                    -   kappa_fc[1](i,j  ,k)*(phiOld(i,j  ,k) - phiOld(i,j-1,k)) )/(dx[1]*dx[1])
#if AMREX_SPACEDIM > 2
                    + ( kappa_fc[2](i,j,k+1)*(phiOld(i,j,k+1) - phiOld(i,j,k  ))
                    -   kappa_fc[2](i,j,k  )*(phiOld(i,j,k  ) - phiOld(i,j,k-1)) )/(dx[2]*dx[2])
#endif
#endif
		    );

            });
        }
	amrex::ParallelDescriptor::Barrier();

        // **********************************
        // INCREMENT
        // **********************************

        // update time
        time = time + dt;

        // copy new solution into old solution
        amrex::MultiFab::Copy(phi_old, phi_new, 0, 0, 1, 0);

        // Tell the I/O Processor to write out which step we're doing
        amrex::Print() << "Advanced step " << step << "\n";

        // **********************************
        // WRITE PLOTFILE AT GIVEN INTERVAL
        // **********************************

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && step%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",step,5);
            WriteSingleLevelPlotfile(pltfile, phi_new, {"phi"}, geom, time, step);
        }
    }

    // Write plot at final time
    if (plot_int > 0 && nsteps%plot_int != 0)
    {
      const std::string& pltfile = amrex::Concatenate("plt",nsteps,5);
      WriteSingleLevelPlotfile(pltfile, phi_new, {"phi"}, geom, time, nsteps);
    }

    Py_DECREF(ac_driver);
    Py_DECREF(py_thermal_properties);

    // Finalize the Python interpreter
    Py_Finalize();

    }
    amrex::Finalize();
    return 0;
}


