use crate::{tri_at, CellType, Index, Mesh, PolyMesh, Real, SortedTri, TetFace};
use std::fmt;

use crate::attrib::{Attrib, AttribDict, IntrinsicAttribute};
use crate::index::CheckedIndex;
use crate::topology::{
    CellIndex, CellVertex, CellVertexIndex, FaceIndex, FaceVertexIndex, NumVertices, VertexIndex,
};
use ahash::AHashMap as HashMap;
use ahash::RandomState;
use flatk::Chunked;

/// A polygon with N sides, whose indices are sorted
#[derive(Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
pub struct SortedNgon {
    // Unfortinately need to use a vec because there is no upper limit on the ngon size
    // in vtk
    pub sorted_indices: Vec<usize>,
}

impl SortedNgon {
    fn new(indices: Vec<usize>) -> Self {
        let mut indices = indices.clone();
        indices.sort();
        SortedNgon {
            sorted_indices: indices,
        }
    }
}

/// A triangle face of a tetrahedron within a `TetMesh`.
#[derive(Clone, Eq)]
pub struct Polygon {
    /// Vertex indices in the source mesh forming this face.
    pub ngon: Vec<usize>,
    /// Index of the corresponding cell within the source tetmesh.
    pub cell_idx: usize,
    /// Starting idx of the face within the cell
    pub start_idx: u16,
    pub cell_type: CellType,
}

impl fmt::Debug for Polygon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Face {{ tri: {:?}, cell_index: {}, face_start_index: {}, cell_type: {:?} }}",
            self.ngon, self.cell_idx, self.start_idx, self.cell_type
        )
    }
}

/// Consider any permutation of the triangle to be equivalent to the original.
impl PartialEq for Polygon {
    fn eq(&self, other: &Polygon) -> bool {
        if other.ngon.clone().sort_unstable() == self.ngon.clone().sort_unstable() {
            true
        } else {
            false
        }
    }
}

/// A utility function to index a slice using four indices, creating a new array of 4
/// corresponding entries of the slice.
fn ngon_at<T>(slice: &[T], ngon: &Vec<usize>) -> Vec<T> {
    ngon.iter().map(|v| slice[*v]).collect::<Vec<_>>()
}

/// A quad with sorted vertices
#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash, Debug)]
struct SortedQuad {
    pub sorted_indices: [usize; 4],
}

impl SortedQuad {
    fn new(indices: [usize; 4]) -> Self {
        let mut indices = indices.clone();
        indices.sort();
        SortedQuad {
            sorted_indices: indices,
        }
    }
}

/// A triangle face of a tetrahedron within a `TetMesh`.
#[derive(Copy, Clone, Eq)]
pub struct QuadFace {
    /// Vertex indices in the source mesh forming this face.
    pub quad: [usize; 4],
    /// Index of the corresponding quad within the source mesh.
    pub cell_index: usize,
    /// Index of the face within the cell
    pub face_index: u16,
    pub cell_type: CellType,
}

impl fmt::Debug for QuadFace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuadFace {{ quad: {:?}, cell_index: {}, face_index: {}, cell_type: {:?} }}",
            self.quad, self.cell_index, self.face_index, self.cell_type
        )
    }
}

impl QuadFace {
    #[rustfmt::skip]
    const PERMUTATIONS: [[usize; 4]; 24] = [
        [1, 2, 3, 4], [2, 1, 3, 4], [3, 1, 2, 4], [1, 3, 2, 4],
        [2, 3, 1, 4], [3, 2, 1, 4], [3, 2, 4, 1], [2, 3, 4, 1],
        [4, 3, 2, 1], [3, 4, 2, 1], [2, 4, 3, 1], [4, 2, 3, 1],
        [4, 1, 3, 2], [1, 4, 3, 2], [3, 4, 1, 2], [4, 3, 1, 2],
        [1, 3, 4, 2], [3, 1, 4, 2], [2, 1, 4, 3], [1, 2, 4, 3],
        [4, 2, 1, 3], [2, 4, 1, 3], [1, 4, 2, 3], [4, 1, 2, 3],
    ];
}

/// A utility function to index a slice using four indices, creating a new array of 4
/// corresponding entries of the slice.
fn quad_at<T: Copy>(slice: &[T], quad: &[usize; 4]) -> [T; 4] {
    [
        slice[quad[0]],
        slice[quad[1]],
        slice[quad[2]],
        slice[quad[3]],
    ]
}

/// Consider any permutation of the triangle to be equivalent to the original.
impl PartialEq for QuadFace {
    fn eq(&self, other: &QuadFace) -> bool {
        for p in Self::PERMUTATIONS.iter() {
            if quad_at(&other.quad, p) == self.quad {
                return true;
            }
        }
        false
    }
}

impl PartialOrd for QuadFace {
    fn partial_cmp(&self, other: &QuadFace) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Lexicographic ordering of the sorted indices.
impl Ord for QuadFace {
    fn cmp(&self, other: &QuadFace) -> std::cmp::Ordering {
        let mut quad = self.quad;
        quad.sort_unstable();
        let mut other_quad = other.quad;
        other_quad.sort_unstable();
        quad.cmp(&other_quad)
    }
}

impl<T: Real> Mesh<T> {
    /// A helper function to compute surface topology of a generic mesh specified by the given cells.
    ///
    /// The algorithm is to iterate over every face and upon seeing a duplicate, remove it from
    /// the list. this will leave only unique faces, which correspond to the surface of the
    /// `Mesh`.
    ///
    /// This function assumes that the given Mesh is a manifold.
    fn surface_ngon_set<'a>(
        indices: &flatk::Clumped<Vec<usize>>,
        types: impl std::iter::ExactSizeIterator<Item = &'a CellType> + Clone,
        poly_faces: &Chunked<Vec<u16>>,
    ) -> (
        HashMap<SortedTri, TetFace>,
        HashMap<SortedQuad, QuadFace>,
        HashMap<u16, HashMap<SortedNgon, Polygon>>,
    ) {
        let mut triangles: HashMap<SortedTri, TetFace> = {
            // This will make surfacing tetmeshes deterministic.
            let hash_builder = RandomState::with_seeds(7, 47, 2377, 719);
            HashMap::with_hasher(hash_builder)
        };
        let mut quads: HashMap<SortedQuad, QuadFace> = {
            let hash_builder = RandomState::with_seeds(7, 47, 2377, 719);
            HashMap::with_hasher(hash_builder)
        };

        let mut poly_maps: HashMap<u16, HashMap<SortedNgon, Polygon>> = HashMap::default();

        //return (triangles, quads);
        for (idx, (cells, cell_type)) in indices.clump_iter().zip(types).enumerate() {
            cell_type.enumerate_faces(
                // The three closures are essentially the same, just removing all duplicates.
                |face_index, tri_face| {
                    for (i, cell) in cells.iter().enumerate() {
                        let face = TetFace {
                            tri: tri_at(cell, tri_face),
                            tet_index: i,
                            face_index: face_index as u16,
                            cell_type: *cell_type,
                        };

                        let key = SortedTri::new(face.tri);

                        if triangles.remove(&key).is_none() {
                            triangles.insert(key, face);
                        }
                    }
                },
                |face_index, quad_face| {
                    for (i, cell) in cells.iter().enumerate() {
                        let face = QuadFace {
                            quad: quad_at(cell, quad_face),
                            cell_index: i,
                            face_index: face_index as u16,
                            cell_type: *cell_type,
                        };

                        let key = SortedQuad::new(face.quad);

                        if quads.remove(&key).is_none() {
                            quads.insert(key, face);
                        }
                    }
                },
                |face_start_index, ngon| {
                    for (i, cell) in cells.iter().enumerate() {
                        let face = Polygon {
                            ngon: ngon_at(cell, ngon),
                            cell_idx: i,
                            start_idx: face_start_index as u16,
                            cell_type: *cell_type,
                        };

                        let key = SortedNgon::new(face.ngon.clone());
                        let ngons = poly_maps.entry(ngon.len() as u16).or_insert({
                            let hash_builder = RandomState::with_seeds(7, 47, 2377, 719);
                            HashMap::with_hasher(hash_builder)
                        });
                        if ngons.remove(&key).is_none() {
                            ngons.insert(key, face);
                        }
                    }
                },
                idx,
                poly_faces,
            );
        }

        (triangles, quads, poly_maps)
    }

    /// Extract the surface ngon information of the `Mesh`.
    ///
    /// Only record those faces that are accepted by `filter`.
    ///
    /// This includes the ngon topology, which cell each ngon came from and which face on
    /// the originating cell it belongs to.  The returned vectors have the same size.
    ///
    /// This function assumes that the given mesh is a manifold.
    ///
    /// (vertices, offsets, cells, cell_face_indices, cell_types)
    pub fn surface_ngon_data<F1, F2, F3>(
        &self,
        tri_filter: F1,
        quad_filter: F2,
        ngon_filter: F3,
        poly_faces: &flatk::Chunked<Vec<u16>>,
    ) -> (
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
        Vec<CellType>,
    )
    where
        F1: FnMut(&TetFace) -> bool,
        F2: FnMut(&QuadFace) -> bool,
        F3: FnMut(u16, &Polygon) -> bool,
    {
        let (triangles, quads, ngons) =
            Self::surface_ngon_set(&self.indices, self.types.iter(), poly_faces);

        let total = triangles.len() + quads.len() + ngons.iter().map(|m| m.1.len()).sum();

        let mut vertices = Vec::new();
        let mut offsets = Vec::with_capacity(total + 1);
        let mut cell_indices = Vec::with_capacity(total);
        let mut cell_face_indices = Vec::with_capacity(total);
        let mut cell_types = Vec::with_capacity(total);

        offsets.push(0); // Initial offset

        for face in triangles
            .into_iter()
            .map(|(_, face)| face)
            .filter(tri_filter)
        {
            vertices.extend_from_slice(&face.tri);
            offsets.push(vertices.len());
            cell_indices.push(face.tet_index);
            cell_face_indices.push(face.face_index.into());
            cell_types.push(face.cell_type);
        }

        for face in quads.into_iter().map(|(_, face)| face).filter(quad_filter) {
            vertices.extend_from_slice(&face.quad);
            offsets.push(vertices.len());
            cell_indices.push(face.cell_index);
            cell_face_indices.push(face.face_index as usize);
            cell_types.push(face.cell_type);
        }

        for (edges, face) in ngons
            .into_iter()
            .flat_map(|(edges, faces)| faces.iter().map(|(_, face)| (edges, face)))
            .filter(ngon_filter)
        {
            vertices.extend_from_slice(&face.ngon);
            offsets.push(vertices.len());
            cell_indices.push(face.cell_idx);
            cell_face_indices.push(face.face_idx as usize);
            cell_types.push(face.cell_type);
        }

        (
            vertices,
            offsets,
            cell_indices,
            cell_face_indices,
            cell_types,
        )
    }

    pub fn surface_mesh(&self) -> PolyMesh<T> {
        self.surface_trimesh_with_mapping_and_filter(None, None, None, None, |_| true, |_| true)
    }

    pub fn surface_trimesh_with_mapping(
        &self,
        original_vertex_index_name: Option<&str>,
        original_tet_index_name: Option<&str>,
        original_tet_vertex_index_name: Option<&str>,
        original_tet_face_index_name: Option<&str>,
    ) -> PolyMesh<T> {
        self.surface_trimesh_with_mapping_and_filter(
            original_vertex_index_name,
            original_tet_index_name,
            original_tet_vertex_index_name,
            original_tet_face_index_name,
            |_| true,
            |_| true,
            (),
        )
    }

    pub fn surface_trimesh_with_mapping_and_filter(
        &self,
        original_vertex_index_name: Option<&str>,
        original_tet_index_name: Option<&str>,
        original_tet_vertex_index_name: Option<&str>,
        original_tet_face_index_name: Option<&str>,
        tri_filter: impl FnMut(&TetFace) -> bool,
        quad_filter: impl FnMut(&QuadFace) -> bool,
        ngon_filter: impl FnMut(&QuadFace) -> bool,
    ) -> PolyMesh<T> {
        // Get the surface topology.
        let (mut topo, mut offsets, cell_indices, cell_face_indices, cell_types) = self
            .surface_ngon_data(
                tri_filter,
                quad_filter,
                ngon_filter,
                &self.polyhedra_face_counts,
            );

        // Record which vertices we have already handled.
        let mut seen = vec![-1isize; self.num_vertices()];

        // Record the mapping back to vertices.
        let mut original_vertex_index = Vec::with_capacity(topo.len());

        // Accumulate surface vertex positions for the new mesh.
        let mut surf_vert_pos = Vec::with_capacity(topo.len());

        for idx in topo.iter_mut() {
            if seen[*idx] == -1 {
                surf_vert_pos.push(self.vertex_positions[*idx]);
                original_vertex_index.push(*idx);
                seen[*idx] = (surf_vert_pos.len() - 1) as isize;
            }
            *idx = seen[*idx] as usize;
        }

        surf_vert_pos.shrink_to_fit();
        original_vertex_index.shrink_to_fit();

        let num_surf_verts = surf_vert_pos.len();

        // Transfer vertex attributes.
        let mut vertex_attributes: AttribDict<VertexIndex> = AttribDict::new();

        for (name, attrib) in self.attrib_dict::<VertexIndex>().iter() {
            let new_attrib = attrib.duplicate_with_len(num_surf_verts, |mut new, old| {
                for (&idx, val) in seen.iter().zip(old.iter()) {
                    if idx != -1 {
                        new.get_mut(idx as usize).clone_from_other(val).unwrap();
                    }
                }
            });
            vertex_attributes.insert(name.to_string(), new_attrib);
        }

        // Transfer face attributes from cell attributes.
        let mut face_attributes: AttribDict<FaceIndex> = AttribDict::new();

        for (name, attrib) in self.attrib_dict::<CellIndex>().iter() {
            face_attributes.insert(
                name.to_string(),
                attrib.promote_with(|new, old| {
                    for &cell_idx in cell_indices.iter() {
                        new.push_cloned(old.get(cell_idx));
                    }
                }),
            );
        }

        // Mapping from face vertex index to its original tet vertex index.
        let mut tet_vertex_index = Vec::new();
        if original_tet_vertex_index_name.is_some() {
            tet_vertex_index.reserve(topo.len());
            for (&cell_idx, &cell_face_idx, cell_type) in cell_indices
                .iter()
                .zip(cell_face_indices.iter())
                .zip(cell_types.iter())
                .map(|((a, b), c)| (a, b, c))
            {
                for &i in cell_type.nth_face_vertices(cell_face_idx) {
                    tet_vertex_index.push(self.cell_vertex(cell_idx, i));
                }
            }
        }

        // Transfer face vertex attributes from tetmesh.
        let mut face_vertex_attributes: AttribDict<FaceVertexIndex> = AttribDict::new();

        for (name, attrib) in self.attrib_dict::<CellVertexIndex>().iter() {
            face_vertex_attributes.insert(
                name.to_string(),
                attrib.promote_with(|new, old| {
                    for (&cell_idx, &cell_face_idx, cell_type) in cell_indices
                        .iter()
                        .zip(cell_face_indices.iter())
                        .zip(cell_types.iter())
                        .map(|((a, b), c)| (a, b, c))
                    {
                        for &i in cell_type.nth_face_vertices(cell_face_idx) {
                            let cell_vtx_idx = self.cell_vertex(cell_idx, i);
                            new.push_cloned(old.get(Index::from(cell_vtx_idx).unwrap()));
                        }
                    }
                }),
            );
        }

        offsets.push(topo.len());
        let mut polymesh = PolyMesh {
            vertex_positions: IntrinsicAttribute::from_vec(surf_vert_pos),
            indices: topo,
            offsets,
            vertex_attributes,
            face_attributes,
            face_vertex_attributes,
            face_edge_attributes: AttribDict::new(), // TetMeshes don't have edge attributes (yet)
            attribute_value_cache: self.attribute_value_cache.clone(),
        };

        // Add the mapping to the original tetmesh. Overwrite any existing attributes.
        if let Some(name) = original_vertex_index_name {
            polymesh
                .set_attrib_data::<_, VertexIndex>(name, original_vertex_index)
                .expect("Failed to add original vertex index attribute.");
        }

        if let Some(name) = original_tet_index_name {
            polymesh
                .set_attrib_data::<_, FaceIndex>(name, cell_indices)
                .expect("Failed to add original tet index attribute.");
        }

        if let Some(name) = original_tet_vertex_index_name {
            polymesh
                .set_attrib_data::<_, FaceVertexIndex>(name, tet_vertex_index)
                .expect("Failed to add original tet vertex index attribute.");
        }

        if let Some(name) = original_tet_face_index_name {
            polymesh
                .set_attrib_data::<_, FaceIndex>(name, cell_face_indices)
                .expect("Failed to add original tet face index attribute.");
        }

        polymesh
    }
}
#[cfg(test)]
mod tests {
    use crate::mesh::{CellType, Mesh};

    #[test]
    fn test_surface_ngon_set() {
        // Create a simple mesh with a tetrahedron and a pyramid
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.0, 1.0],
        ];
        let cells = vec![
            vec![3, 2, 4, 5],    // tetrahedron
            vec![0, 1, 4, 2, 3], // pyramid
        ];
        // tetrahedron on top of pyramid
        // (just imagine the line from 5 to 4 for the tetrahedrons last side)
        //
        //        2.+------+4
        //        //|    / |
        //       / ||   /  |
        //      / / |  /   |
        //     / / 0+-/----+1
        //   5+  | / /   -/
        //    | / / /  -/
        //    |/ / / -/
        //    ||//--/
        //    ///-/
        //   3+-/

        let types = vec![CellType::Tetrahedron, CellType::Pyramid];

        let mesh = Mesh::from_cells_and_types(points, cells, types);

        let (triangles, quads) = Mesh::<f64>::surface_ngon_set(&mesh.indices, mesh.types.iter());

        triangles
            .iter()
            .for_each(|(key, value)| println!("{:?}: {:?}", key, value));
        quads
            .iter()
            .for_each(|(key, value)| println!("{:?}: {:?}", key, value));

        // assert that the triangle connecting the two shapes doesn't exist.
        assert!(triangles
            .iter()
            .find(|x| x.0.sorted_indices == [2, 3, 4])
            .is_none())
    }
}
