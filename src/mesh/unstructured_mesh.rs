//!
//! Unstructured mesh module.
//!
//! This module defines a mesh data structure cunstructed from unstructured
//! cells of arbitrary shape.
//!

use crate::attrib::*;
use crate::mesh::topology::*;
use crate::mesh::vertex_positions::VertexPositions;
use crate::utils::slice::apply_permutation_with_seen;
use crate::Real;

use flatk::*;

/// A marker for the type of cell contained in a Mesh.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CellType {
    Tetrahedron,
    Triangle,
}

impl CellType {
    /// Returns the number of vertices referenced by this cell type.
    pub fn num_verts(&self) -> usize {
        match self {
            CellType::Triangle => 3,
            CellType::Tetrahedron => 4,
        }
    }
}

/// Mesh with arbitrarily shaped elements or cells.
///
/// The currently supported cell types are listed in the [`CellType`] enum.
///
/// NOTE: We stick with the terminology cell but these could very well be called
/// elements. The exact terminology would depend on how this mesh is used.
#[derive(Clone, Debug, PartialEq, Attrib, Intrinsic)]
pub struct Mesh<T: Real> {
    /// Vertex positions intrinsic attribute.
    #[intrinsic(VertexPositions)]
    pub vertex_positions: IntrinsicAttribute<[T; 3], VertexIndex>,
    /// Indices into `vertices`. Each chunk represents a cell, and cells
    /// can have an arbitrary number of referenced vertices. Each clump
    /// represents cells of the same kind.
    pub indices: flatk::Clumped<Vec<usize>>,
    /// Types of cells, one for each clump in `indices`.
    pub types: Vec<CellType>,
    /// Vertex attributes.
    pub vertex_attributes: AttribDict<VertexIndex>,
    /// Cell attributes.
    pub cell_attributes: AttribDict<CellIndex>,
    /// Cell vertex attributes.
    pub cell_vertex_attributes: AttribDict<CellVertexIndex>,
    /// Indirect attribute value cache
    pub attribute_value_cache: AttribValueCache,
}

impl<T: Real> Mesh<T> {
    /// Constructs a `Mesh` from an array of vertices and nested array of cells with an
    /// accompanying array of cell types with the same size.
    ///
    /// Each element of the `cells` array is a contiguous vector of vertex indices into the given
    /// `verts` array for each cell type found in `types`.
    ///
    /// # Examples
    /// ```
    /// use meshx::mesh::{Mesh, CellType};
    /// let points = vec![
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [1.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    ///     [1.0, 0.0, 1.0]];
    /// let cells = vec![
    ///     vec![
    ///         0, 1, 2, // first triangle
    ///         1, 3, 2, // second triangle
    ///     ],
    ///     vec![0, 1, 5, 4], // tetrahedron
    /// ];
    /// let types = vec![CellType::Triangle, CellType::Tetrahedron];
    ///
    /// let mesh = Mesh::from_cells_and_types(points, cells, types);
    ///
    /// assert_eq!(mesh.indices.data, vec![0, 1, 2, 1, 3, 2, 0, 1, 5, 4]);
    /// assert_eq!(mesh.types, vec![CellType::Triangle, CellType::Tetrahedron]);
    /// let mut iter = mesh.cell_iter();
    /// assert_eq!(iter.next(), Some(&[0,1,2][..]));
    /// assert_eq!(iter.next(), Some(&[1,3,2][..]));
    /// assert_eq!(iter.next(), Some(&[0,1,5,4][..]));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn from_cells_and_types(
        verts: impl Into<Vec<[T; 3]>>,
        cells: impl Into<Vec<Vec<usize>>>,
        types: impl Into<Vec<CellType>>,
    ) -> Mesh<T> {
        Self::from_cells_and_types_impl(verts.into(), cells.into(), types.into())
    }

    // A non-generic implementation of the `from_cells_and_types` constructor.
    fn from_cells_and_types_impl(
        verts: Vec<[T; 3]>,
        cells: Vec<Vec<usize>>,
        types: Vec<CellType>,
    ) -> Mesh<T> {
        assert_eq!(cells.len(), types.len());
        let sizes: Vec<_> = types.iter().map(CellType::num_verts).collect();
        let counts: Vec<_> = sizes
            .iter()
            .zip(cells.iter())
            .map(|(s, c)| c.len() / s)
            .collect();
        let cells = cells.into_iter().flatten().collect();
        let clumped_indices = flatk::Clumped::from_sizes_and_counts(sizes, counts, cells);

        Mesh {
            vertex_positions: IntrinsicAttribute::from_vec(verts),
            indices: clumped_indices,
            types,
            vertex_attributes: AttribDict::new(),
            cell_attributes: AttribDict::new(),
            cell_vertex_attributes: AttribDict::new(),
            attribute_value_cache: AttribValueCache::with_hasher(Default::default()),
        }
    }

    /// Constructs a `Mesh` from an array of vertices and cells with a number of counts of each cell type
    /// appearing in `cells` given by `counts` and `types`.
    ///
    /// The `cells` array contains contiguous indices into the vertex array for each cell.
    ///
    /// # Examples
    /// ```
    /// use meshx::mesh::{Mesh, CellType};
    /// let points = vec![
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [1.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    ///     [1.0, 0.0, 1.0]];
    /// let cells = vec![0, 1, 2, // first triangle
    ///                  1, 3, 2, // second triangle
    ///                  0, 1, 5, 4]; // tetrahedron
    /// let counts = vec![2, 1];
    /// let types = vec![CellType::Triangle, CellType::Tetrahedron];
    ///
    /// let mesh = Mesh::from_cells_counts_and_types(points, cells, counts, types);
    ///
    /// assert_eq!(mesh.indices.data, vec![0, 1, 2, 1, 3, 2, 0, 1, 5, 4]);
    /// assert_eq!(mesh.types, vec![CellType::Triangle, CellType::Tetrahedron]);
    /// let mut iter = mesh.cell_iter();
    /// assert_eq!(iter.next(), Some(&[0,1,2][..]));
    /// assert_eq!(iter.next(), Some(&[1,3,2][..]));
    /// assert_eq!(iter.next(), Some(&[0,1,5,4][..]));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn from_cells_counts_and_types(
        verts: impl Into<Vec<[T; 3]>>,
        cells: impl Into<Vec<usize>>,
        counts: impl AsRef<[usize]>,
        types: impl Into<Vec<CellType>>,
    ) -> Mesh<T> {
        Self::from_cells_counts_and_types_impl(
            verts.into(),
            cells.into(),
            counts.as_ref(),
            types.into(),
        )
    }

    // Non-generic implementation of the `from_cells_counts_and_types` method.
    fn from_cells_counts_and_types_impl(
        verts: Vec<[T; 3]>,
        cells: Vec<usize>,
        counts: &[usize],
        types: Vec<CellType>,
    ) -> Mesh<T> {
        let sizes: Vec<_> = types.iter().map(CellType::num_verts).collect();
        let clumped_indices = flatk::Clumped::from_sizes_and_counts(sizes, counts, cells);

        Mesh {
            vertex_positions: IntrinsicAttribute::from_vec(verts),
            indices: clumped_indices,
            types,
            vertex_attributes: AttribDict::new(),
            cell_attributes: AttribDict::new(),
            cell_vertex_attributes: AttribDict::new(),
            attribute_value_cache: AttribValueCache::with_hasher(Default::default()),
        }
    }

    /// Constructs a `Mesh` from an array of vertices and clumped cells with
    /// associated types.
    ///
    /// The `cells` array contains contiguous indices into the vertex array for
    /// each cell.  The `offsets` array contains offsets into the `cells` array,
    /// beginning with 0 and ending with the number of elements in the `cells`
    /// array.  Each additional offset in the middle should index each time the
    /// type of cell changes inside `cells`.  The `types` array contains the
    /// cell types for each contiguous _block_ of cells, where each block has
    /// the same type.  For instance if cells contains 10 triangles and 4
    /// tetrahedra, then `offests` should contain `[0, 10, 14]` and `types`
    /// should contain two elements: `[CellType::Triangle,
    /// CellType::Tetrahedron]`.
    ///
    /// # Examples
    /// ```
    /// use meshx::mesh::{Mesh, CellType};
    /// let points = vec![
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [1.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    ///     [1.0, 0.0, 1.0]];
    /// let cells = vec![0, 1, 2, // first triangle
    ///                  1, 3, 2, // second triangle
    ///                  0, 1, 5, 4]; // tetrahedron
    /// let offsets = vec![0, 2, 3];
    /// let types = vec![CellType::Triangle, CellType::Tetrahedron];
    ///
    /// let mesh = Mesh::from_clumped_cells_and_types(points, cells, offsets, types);
    ///
    /// assert_eq!(mesh.indices.data, vec![0, 1, 2, 1, 3, 2, 0, 1, 5, 4]);
    /// assert_eq!(mesh.types, vec![CellType::Triangle, CellType::Tetrahedron]);
    /// let mut iter = mesh.cell_iter();
    /// assert_eq!(iter.next(), Some(&[0,1,2][..]));
    /// assert_eq!(iter.next(), Some(&[1,3,2][..]));
    /// assert_eq!(iter.next(), Some(&[0,1,5,4][..]));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn from_clumped_cells_and_types(
        verts: impl Into<Vec<[T; 3]>>,
        cells: impl Into<Vec<usize>>,
        offsets: impl Into<Vec<usize>>,
        types: impl Into<Vec<CellType>>,
    ) -> Mesh<T> {
        Self::from_clumped_cells_and_types_impl(
            verts.into(),
            cells.into(),
            offsets.into(),
            types.into(),
        )
    }

    // A non-generic implementation of `from_clumped_cells_and_types`.
    fn from_clumped_cells_and_types_impl(
        verts: Vec<[T; 3]>,
        cells: Vec<usize>,
        mut chunk_offsets: Vec<usize>,
        types: Vec<CellType>,
    ) -> Mesh<T> {
        // Make sure offsets is correctly structured (always contains a 0 as required by `from_clumped_offsets` below).
        if chunk_offsets.is_empty() {
            chunk_offsets.push(0);
        }

        let offsets: Vec<_> = chunk_offsets
            .iter()
            .enumerate()
            .scan((0, 0), |(prev_off, prev_chunk_off), (i, &chunk_off)| {
                Some(if i > 0 {
                    *prev_off += (chunk_off - *prev_chunk_off) * types[i - 1].num_verts();
                    *prev_chunk_off = chunk_off;
                    *prev_off
                } else {
                    0
                })
            })
            .collect();

        Mesh {
            vertex_positions: IntrinsicAttribute::from_vec(verts),
            indices: flatk::Clumped::from_clumped_offsets(chunk_offsets, offsets, cells),
            types,
            vertex_attributes: AttribDict::new(),
            cell_attributes: AttribDict::new(),
            cell_vertex_attributes: AttribDict::new(),
            attribute_value_cache: AttribValueCache::with_hasher(Default::default()),
        }
    }

    /// Constructs a `Mesh` from an array of vertices and cells with associated types given queried with a function.
    ///
    /// The `cells` array contains the indices into the vertex array for each cell preceeded by the
    /// number of vertices in the corresponding cell. I.e. `cells` is expected to be structured as
    /// a contiguous array of a number (corresponding to the number of vertices in the cell)
    /// followed by the vertex indices (in the same cell):
    /// ```verbatim
    ///     n i_1 i_2 ... i_n m j_1 j_2 ... j_m ...
    /// ```
    ///
    /// # Examples
    /// ```
    /// use meshx::mesh::{Mesh, CellType};
    /// let points = vec![
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [1.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    ///     [1.0, 0.0, 1.0]];
    /// let cells = vec![3, 0, 1, 2, // first triangle
    ///                  3, 1, 3, 2, // second triangle
    ///                  4, 0, 1, 5, 4]; // tetrahedron
    ///
    /// let mesh = Mesh::from_cells_with_type(points, cells, |i| if i < 2 { CellType::Triangle } else { CellType::Tetrahedron });
    ///
    /// assert_eq!(mesh.indices.data, vec![0, 1, 2, 1, 3, 2, 0, 1, 5, 4]);
    /// assert_eq!(mesh.types, vec![CellType::Triangle, CellType::Tetrahedron]);
    /// let mut iter = mesh.cell_iter();
    /// assert_eq!(iter.next(), Some(&[0,1,2][..]));
    /// assert_eq!(iter.next(), Some(&[1,3,2][..]));
    /// assert_eq!(iter.next(), Some(&[0,1,5,4][..]));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn from_cells_with_type(
        verts: impl Into<Vec<[T; 3]>>,
        cells: impl AsRef<[usize]>,
        type_at: impl Fn(usize) -> CellType,
    ) -> Mesh<T> {
        Self::from_cells_with_type_impl(verts.into(), cells.as_ref(), type_at)
    }

    // A mostly non-generic implementation of `from_cells_with_type`.
    fn from_cells_with_type_impl(
        verts: Vec<[T; 3]>,
        cells: &[usize],
        type_at: impl Fn(usize) -> CellType,
    ) -> Mesh<T> {
        let mut indices = Vec::new();
        let mut offsets = Vec::new();
        let mut i = 0;
        while i < cells.len() {
            let n = cells[i];
            offsets.push(indices.len());
            i += 1;

            for k in 0..n {
                indices.push(cells[i + k]);
            }

            i += n;
        }

        offsets.push(indices.len());

        let chunked_indices = flatk::Chunked::from_offsets(offsets, indices);

        let clumped_indices = flatk::Clumped::from(chunked_indices);

        let types = clumped_indices
            .chunks
            .chunk_offsets
            .iter()
            .map(type_at)
            .take(clumped_indices.chunks.num_clumps())
            .collect();

        Mesh {
            vertex_positions: IntrinsicAttribute::from_vec(verts),
            indices: clumped_indices,
            types,
            vertex_attributes: AttribDict::new(),
            cell_attributes: AttribDict::new(),
            cell_vertex_attributes: AttribDict::new(),
            attribute_value_cache: AttribValueCache::with_hasher(Default::default()),
        }
    }

    /// Returns an iterator over immutable index slices for each cell in this mesh.
    pub fn cell_iter(&self) -> impl ExactSizeIterator<Item = &[usize]> {
        self.indices.iter()
    }

    /// Returns an iterator over mutable index slices for each cell in this mesh.
    pub fn cell_iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut [usize]> {
        self.indices.iter_mut()
    }

    /// Returns an iterator over cell types for each individual cell of this mesh.
    pub fn cell_type_iter(&self) -> impl Iterator<Item = CellType> + '_ {
        self.types
            .iter()
            .zip(self.indices.chunks.chunk_offsets.sizes())
            .flat_map(|(&ty, n)| std::iter::repeat(ty).take(n))
    }

    /// Reverse the order of each cell in this mesh.
    #[inline]
    pub fn reverse(&mut self) {
        for cell in self.cell_iter_mut() {
            cell.reverse();
        }

        let Self {
            cell_vertex_attributes,
            indices,
            ..
        } = self;

        // TODO: Consider doing reversing lazily using a flag field.

        // Since each vertex has an associated cell vertex attribute, we must remap those
        // as well.
        // Reverse cell vertex attributes
        for (_, attrib) in cell_vertex_attributes.iter_mut() {
            let mut data_slice = attrib.data_mut_slice();
            for cell in indices.iter() {
                let mut i = 0usize;
                let num_verts = cell.len();
                while i < num_verts / 2 {
                    data_slice.swap(cell[i], cell[num_verts - i - 1]);
                    i += 1;
                }
            }
        }
    }

    /// Reverse the order of each cell in this mesh.
    #[inline]
    pub fn reverse_if<F>(&mut self, predicate: F)
    where
        F: Fn(&[usize], CellType) -> bool,
    {
        let Self {
            cell_vertex_attributes,
            indices,
            types,
            ..
        } = self;

        // We need this small clone to break simultaneous mutable borrow.
        let chunk_offsets = indices.chunks.chunk_offsets.clone();

        // This monstrocity is all to satisfy the borrow checker.
        // TODO: Low priority: figure a better way to do this.
        fn cell_iter<'a, F>(
            indices: ClumpedView<'a, &'a mut [usize]>,
            types: &'a [CellType],
            chunk_offsets: &'a Offsets,
            predicate: &'a F,
        ) -> impl Iterator<Item = &'a mut [usize]> + 'a
        where
            F: Fn(&[usize], CellType) -> bool + 'a,
        {
            indices
                .into_iter()
                .zip(
                    types
                        .iter()
                        .zip(chunk_offsets.sizes())
                        .flat_map(|(&ty, n)| std::iter::repeat(ty).take(n)),
                )
                .filter(move |(cell, cell_type)| predicate(*cell, *cell_type))
                .map(|(cell, _)| cell)
        }

        for cell in cell_iter(
            indices.view_mut(),
            types.as_mut_slice(),
            &chunk_offsets,
            &predicate,
        ) {
            cell.reverse();
        }

        // TODO: Consider doing reversing lazily using a flag field.

        // Since each vertex has an associated cell vertex attribute, we must remap those
        // as well.
        // Reverse cell vertex attributes
        for (_, attrib) in cell_vertex_attributes.iter_mut() {
            let mut data_slice = attrib.data_mut_slice();
            for cell in cell_iter(
                indices.view_mut(),
                types.as_mut_slice(),
                &chunk_offsets,
                &predicate,
            ) {
                let mut i = 0usize;
                let num_verts = cell.len();
                while i < num_verts / 2 {
                    data_slice.swap(cell[i], cell[num_verts - i - 1]);
                    i += 1;
                }
            }
        }
    }

    /// Reverse the order of each cell in this mesh.
    ///
    ///  This is the consuming version of the `reverse` method.
    #[inline]
    pub fn reversed(mut self) -> Mesh<T> {
        self.reverse();
        self
    }

    /// Sort vertices by the given key values, and return the reulting order (permutation).
    pub fn sort_vertices_by_key<K, F>(&mut self, mut f: F) -> Vec<usize>
    where
        F: FnMut(usize) -> K,
        K: Ord,
    {
        // Early exit.
        if self.num_vertices() == 0 {
            return Vec::new();
        }

        let num = self.attrib_size::<VertexIndex>();
        debug_assert!(num > 0);

        // Original vertex indices.
        let mut order: Vec<usize> = (0..num).collect();

        // Sort vertex indices by the given key.
        order.sort_by_key(|k| f(*k));

        // Now sort all mesh data according to the sorting given by order.

        let Mesh {
            ref mut vertex_positions,
            ref mut indices,
            ref mut vertex_attributes,
            .. // cell and cell_vertex attributes are unchanged
        } = *self;

        let mut seen = vec![false; vertex_positions.len()];

        // Apply the order permutation to vertex_positions in place
        apply_permutation_with_seen(&order, vertex_positions.as_mut_slice(), &mut seen);

        // Apply permutation to each vertex attribute
        for (_, attrib) in vertex_attributes.iter_mut() {
            let mut data_slice = attrib.data_mut_slice();

            // Clear seen
            seen.iter_mut().for_each(|b| *b = false);

            apply_permutation_with_seen(&order, &mut data_slice, &mut seen);
        }

        // Build a reverse mapping for convenience.
        let mut new_indices = vec![0; order.len()];
        for (new_idx, &old_idx) in order.iter().enumerate() {
            new_indices[old_idx] = new_idx;
        }

        // Remap cell vertices.
        for cell in indices.iter_mut() {
            for vtx_idx in cell.iter_mut() {
                *vtx_idx = new_indices[*vtx_idx];
            }
        }

        order
    }
}

impl<T: Real> Default for Mesh<T> {
    /// Produce an empty mesh.
    ///
    /// This is not particularly useful on its own, however it can be used as a
    /// null case for various mesh algorithms.
    ///
    /// This function allocates two `Vec`s of size 1.
    fn default() -> Self {
        Mesh::from_clumped_cells_and_types(vec![], vec![], vec![0], vec![])
    }
}

impl<T: Real> NumVertices for Mesh<T> {
    #[inline]
    fn num_vertices(&self) -> usize {
        self.vertex_positions.len()
    }
}

impl<T: Real> NumCells for Mesh<T> {
    #[inline]
    fn num_cells(&self) -> usize {
        self.indices.len()
    }
}

impl<T: Real> CellVertex for Mesh<T> {
    #[inline]
    fn vertex<CVI>(&self, cv_idx: CVI) -> VertexIndex
    where
        CVI: Copy + Into<CellVertexIndex>,
    {
        let cv_idx = usize::from(cv_idx.into());
        debug_assert!(cv_idx < self.num_cell_vertices());
        self.indices.data[cv_idx].into()
    }

    #[inline]
    fn cell_vertex<CI>(&self, cidx: CI, which: usize) -> Option<CellVertexIndex>
    where
        CI: Copy + Into<CellIndex>,
    {
        let cidx = usize::from(cidx.into());
        let num_verts_at_cell = self.indices.view().get(cidx)?.len();
        if which >= num_verts_at_cell {
            None
        } else {
            // SAFETY: cidx is known to be less than num_offsets - 1 from the
            // `num_verts_at_cell` computation.
            // We actually only need cidx to be less than num_offsets for safety.
            Some((unsafe { self.indices.chunks.offset_value_unchecked(cidx) } + which).into())
        }
    }

    #[inline]
    fn num_cell_vertices(&self) -> usize {
        self.indices.data.len()
    }

    #[inline]
    fn num_vertices_at_cell<CI>(&self, cidx: CI) -> usize
    where
        CI: Copy + Into<CellIndex>,
    {
        let cidx = usize::from(cidx.into());
        self.indices.view().at(cidx).len()
    }
}

impl<T: Real> From<super::TriMesh<T>> for Mesh<T> {
    /// Convert a triangle mesh into an unstructured mesh.
    fn from(mesh: super::TriMesh<T>) -> Mesh<T> {
        let super::TriMesh {
            vertex_positions,
            indices,
            vertex_attributes,
            face_attributes,
            face_vertex_attributes,
            .. // Missing face-edge and vertex-face topo attributes
        } = mesh;

        let types = vec![CellType::Triangle; 1];

        // The cell and cell-vertex topology is the same as the original face
        // and face-vertex topology, so we just need to move all the attributes
        // as is.

        let mut cell_attributes = AttribDict::new();
        for (name, attrib) in face_attributes {
            let mut new_attrib = attrib.promote_empty::<CellIndex>();
            new_attrib.data = attrib.data;
            cell_attributes.insert(name, new_attrib);
        }

        let mut cell_vertex_attributes = AttribDict::new();
        for (name, attrib) in face_vertex_attributes {
            let mut new_attrib = attrib.promote_empty::<CellVertexIndex>();
            new_attrib.data = attrib.data;
            cell_vertex_attributes.insert(name, new_attrib);
        }

        Mesh {
            vertex_positions,
            indices: flatk::Clumped::from_clumped_offsets(
                vec![0, indices.len()],
                vec![0, 3 * indices.len()],
                flatk::Chunked3::from_array_vec(indices.into_vec()).into_inner(),
            ),
            types,
            vertex_attributes,
            cell_attributes,
            cell_vertex_attributes,
            attribute_value_cache: AttribValueCache::default(),
        }
    }
}

impl<T: Real> From<super::TetMesh<T>> for Mesh<T> {
    /// Convert a tetmesh into an unstructured mesh.
    fn from(mesh: super::TetMesh<T>) -> Mesh<T> {
        let super::TetMesh {
            vertex_positions,
            indices,
            vertex_attributes,
            cell_attributes,
            cell_vertex_attributes,
            ..
        } = mesh;

        let types = vec![CellType::Tetrahedron; 1];

        Mesh {
            vertex_positions,
            indices: flatk::Clumped::<Vec<usize>>::from_clumped_offsets(
                vec![0, indices.len()],
                vec![0, 4 * indices.len()],
                flatk::Chunked4::from_array_vec(indices.into_vec()).into_inner(),
            ),
            types,
            vertex_attributes,
            cell_attributes,
            cell_vertex_attributes,
            attribute_value_cache: AttribValueCache::default(),
        }
    }
}

/// Convert a point cloud into a mesh.
impl<T: Real> From<super::PointCloud<T>> for Mesh<T> {
    fn from(mesh: super::PointCloud<T>) -> Mesh<T> {
        let super::PointCloud {
            vertex_positions,
            vertex_attributes,
        } = mesh;

        Mesh {
            vertex_attributes,
            ..Mesh::from_clumped_cells_and_types(vertex_positions, vec![], vec![0], vec![])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Index;

    fn build_simple_mesh() -> Mesh<f64> {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ];
        let cells = vec![
            3, 0, 1, 2, // first triangle
            3, 1, 3, 2, // second triangle
            4, 0, 1, 5, 4, // tetrahedron
        ];

        Mesh::from_cells_with_type(points, &cells, |i| {
            if i < 2 {
                CellType::Triangle
            } else {
                CellType::Tetrahedron
            }
        })
    }

    #[test]
    fn mesh_test() {
        let mesh = build_simple_mesh();
        assert_eq!(mesh.num_vertices(), 6);
        assert_eq!(mesh.num_cells(), 3);
        assert_eq!(mesh.num_cell_vertices(), 10);

        assert_eq!(Index::from(mesh.cell_to_vertex(1, 1)), 3);
        assert_eq!(Index::from(mesh.cell_to_vertex(0, 2)), 2);
        assert_eq!(mesh.types, vec![CellType::Triangle, CellType::Tetrahedron]);

        assert_eq!(mesh.indices.view().at(0), &[0, 1, 2][..]);
        assert_eq!(mesh.indices.view().at(1), &[1, 3, 2][..]);
        assert_eq!(mesh.indices.view().at(2), &[0, 1, 5, 4][..]);
    }

    #[test]
    fn reverse_only_triangles() {
        let mut mesh = build_simple_mesh();
        mesh.reverse_if(|_, cell_type| matches!(cell_type, CellType::Triangle));
        let mut cell_iter = mesh.cell_iter();
        assert_eq!(cell_iter.next(), Some(&[2, 1, 0][..]));
        assert_eq!(cell_iter.next(), Some(&[2, 3, 1][..]));
        assert_eq!(cell_iter.next(), Some(&[0, 1, 5, 4][..]));
        assert_eq!(cell_iter.next(), None);
    }

    fn sample_points() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    }

    #[test]
    fn from_trimesh_test() {
        use crate::mesh::TriMesh;
        let points = sample_points();
        let tri_faces = vec![
            3, 0, 1, 2, // first triangle
            3, 1, 3, 2, // second triangle
        ];

        let trimesh = TriMesh::new(points.clone(), vec![[0, 1, 2], [1, 3, 2]]);
        let tri_mesh =
            Mesh::from_cells_with_type(points.clone(), &tri_faces, |_| CellType::Triangle);
        assert_eq!(Mesh::from(trimesh), tri_mesh);
    }

    #[test]
    fn from_tetmesh_test() {
        use crate::mesh::TetMesh;
        let points = sample_points();
        let tets = vec![
            4, 0, 1, 3, 2, // just one tet
        ];

        let tetmesh = TetMesh::new(points.clone(), vec![[0, 1, 3, 2]]);
        let tet_mesh = Mesh::from_cells_with_type(points, &tets, |_| CellType::Tetrahedron);
        assert_eq!(Mesh::from(tetmesh), tet_mesh);
    }

    #[test]
    fn from_pointcloud_test() {
        use crate::mesh::PointCloud;
        let points = sample_points();

        let ptcld = PointCloud::new(points.clone());
        let ptcld_mesh = Mesh::from_cells_with_type(
            points.clone(),
            &[],
            |_| CellType::Triangle, /* never called */
        );
        assert_eq!(Mesh::from(ptcld), ptcld_mesh);
    }
}
