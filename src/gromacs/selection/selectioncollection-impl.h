/*
 *
 *                This source code is part of
 *
 *                 G   R   O   M   A   C   S
 *
 *          GROningen MAchine for Chemical Simulations
 *
 * Written by David van der Spoel, Erik Lindahl, Berk Hess, and others.
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2009, The GROMACS development team,
 * check out http://www.gromacs.org for more information.

 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * If you want to redistribute modifications, please consider that
 * scientific software is very special. Version control is crucial -
 * bugs must be traceable. We will be happy to consider code for
 * inclusion in the official distribution, but derived work must not
 * be called official GROMACS. Details are found in the README & COPYING
 * files - if they are missing, get the official version at www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the papers on the package - you can find them in the top README file.
 *
 * For more info, check our website at http://www.gromacs.org
 */
/*! \internal \file
 * \brief
 * Declares private implementation class for gmx::SelectionCollection.
 *
 * This header also defines ::gmx_ana_selcollection_t, which is used in the old
 * C code for handling selection collections.
 *
 * \author Teemu Murtola <teemu.murtola@cbr.su.se>
 * \ingroup module_selection
 */
#ifndef GMX_SELECTION_SELECTIONCOLLECTION_IMPL_H
#define GMX_SELECTION_SELECTIONCOLLECTION_IMPL_H

#include <string>
#include <vector>

#include <typedefs.h>

#include "../options/options.h"
#include "indexutil.h"
#include "selectioncollection.h"

namespace gmx
{
class Selection;
}

/*! \internal
 * \brief
 * Information for a collection of selections.
 *
 * The functions to deal with the structure are defined in selection.h.
 * The structure is allocated with gmx_ana_selcollection_create() and
 * freed with gmx_ana_selcollection_free().
 * Some default values must then be set with
 * gmx_ana_selcollection_set_refpostype() and
 * gmx_ana_selcollection_set_outpostype().
 *
 * After setting the default values, one or more selections can be parsed
 * with gmx_ana_selcollection_parse_*().
 * At latest at this point, the topology must be set with
 * gmx_ana_selcollection_set_topology() unless
 * gmx_ana_selcollection_requires_top() returns FALSE.
 * Once all selections are parsed, they must be compiled all at once using
 * gmx_ana_selcollection_compile().
 * After these calls, gmx_ana_selcollection_get_count() and 
 * gmx_ana_selcollection_get_selections() can be used
 * to get the compiled selections.
 * gmx_ana_selcollection_evaluate() can be used to update the selections for a
 * new frame.
 * gmx_ana_selcollection_evaluate_fin() can be called after all the frames have
 * been processed to restore the selection values back to the ones they were
 * after gmx_ana_selcollection_compile(), i.e., dynamic selections have the
 * maximal index group as their value.
 *
 * At any point, gmx_ana_selcollection_requires_top() can be called to see
 * whether the information provided so far requires loading the topology.
 * gmx_ana_selcollection_print_tree() can be used to print the internal
 * representation of the selections (mostly useful for debugging).
 */
struct gmx_ana_selcollection_t
{
    /** Default reference position type for selections. */
    const char                 *rpost;
    /** Default output position type for selections. */
    const char                 *spost;
    /** TRUE if \ref POS_MASKONLY should be used for output position evaluation. */
    bool                        bMaskOnly;
    /** TRUE if velocities should be evaluated for output positions. */
    bool                        bVelocities;
    /** TRUE if forces should be evaluated for output positions. */
    bool                        bForces;

    /** Root of the selection element tree. */
    struct t_selelem           *root;
    /** Array of compiled selections. */
    std::vector<gmx::Selection *>  sel;
    /** Number of variables defined. */
    int                            nvars;
    /** Selection strings for variables. */
    char                         **varstrs;

    /** Topology for the collection. */
    t_topology                    *top;
    /** Index group that contains all the atoms. */
    struct gmx_ana_index_t         gall;
    /** Position calculation collection used for selection position evaluation. */
    struct gmx_ana_poscalc_coll_t *pcc;
    /** Memory pool used for selection evaluation. */
    struct gmx_sel_mempool_t      *mempool;
    /** Parser symbol table. */
    struct gmx_sel_symtab_t     *symtab;
};

namespace gmx
{

class SelectionCollection::Impl
{
    public:
        typedef std::vector<Selection *> SelectionList;

        enum Flag
        {
            efOwnPositionCollection = 1<<0
        };

        Impl(gmx_ana_poscalc_coll_t *pcc);
        ~Impl();

        bool hasFlag(Flag flag) const { return _flags & flag; }
        void setFlag(Flag flat, bool bSet);
        void clearSymbolTable();
        int registerDefaultMethods();
        int runParser(void *scanner, int maxnr,
                      std::vector<Selection *> *output);

        gmx_ana_selcollection_t _sc;
        Options                 _options;
        std::string             _rpost;
        std::string             _spost;
        int                     _debugLevel;
        unsigned long           _flags;
        gmx_ana_indexgrps_t    *_grps;
};

} // namespace gmx

/* In compiler.c */
/** Prepares the selections for evaluation and performs some optimizations. */
int
gmx_ana_selcollection_compile(gmx::SelectionCollection *coll);

/* In evaluate.c */
/** Evaluates the selection. */
int
gmx_ana_selcollection_evaluate(gmx_ana_selcollection_t *sc,
                               t_trxframe *fr, t_pbc *pbc);
/** Evaluates the largest possible index groups from dynamic selections. */
int
gmx_ana_selcollection_evaluate_fin(gmx_ana_selcollection_t *sc, int nframes);

#endif
