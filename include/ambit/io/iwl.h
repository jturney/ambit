/*
 *  Copyright (C) 2013  Justin Turney
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.

 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.

 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-151 USA.
 */

#if !defined(TENSOR_IO_IWL)
#define TENSOR_IO_IWL

#include "file.h"

#include <ambit/tensor.h>

namespace ambit
{
namespace io
{

namespace details
{

/// The number of integrals per batch to be read in.
static constexpr int integrals_per_buffer__ = 2980;
/// The label in the integral file to use. Should probably be abstracted away
/// but this is what PSI3/4 uses.
static constexpr const char *buffer_key__ = "IWL Buffers";
}

struct IWL : public File
{
    /**
    * This version uses an existing file object to create an iwl object.
    * The act of using this constructor is that the file object you pass is
    * moved into
    * the file parent class of the new iwl object. You do not own the file
    * object you
    * passed in anymore.
    * @param f File object to take ownership of
    * @param cutoff Numerical zero
    * @param psi34_compatible Follow Psi3/4 nomenclature?
    */
    IWL(File &&f, double cutoff = numerical_zero__,
        bool psi34_compatible = true);

    /**
    * This version does not use the file manager class to determine the location
    * of the scratch file. When running in parallel you will most likely open
    * the wrong file.
    * @param full_pathname The full path of the file to open
    * @param om Open mode
    * @param dm Close/Delete mode
    * @param cutoff numerical zero
    * @param psi34_compatible Follow Psi3/4 nomenclature?
    */
    IWL(const std::string &full_pathname, enum OpenMode om,
        enum DeleteMode dm = kDeleteModeKeepOnClose,
        double cutoff = numerical_zero__, bool psi34_compatible = true);

    virtual ~IWL();

    /// number of integrals valid in values, p, q, r, and s.
    const int nintegral;

    /// is this the last buffer?
    const int last_buffer;

    // implements SOA.
    std::vector<double> values;
    std::vector<short int> p;
    std::vector<short int> q;
    std::vector<short int> r;
    std::vector<short int> s;

    /// fetch the next batch of integrals.
    void fetch();

    static void read_one(File &io, const std::string &label, Tensor &tensor);

    static void read_two(IWL &io, Tensor &tensor);

  private:
    /// psi3/4 compatible label structure.
    std::vector<short int> labels_;

    const bool psi34_compatible_;
    const double cutoff_;

    Address read_position_;
};
}
}

#endif
