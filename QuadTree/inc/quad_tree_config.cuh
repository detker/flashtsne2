#ifndef QUAD_TREE_CONFIG
#define QUAD_TREE_CONFIG

#define QUAD_TREE_MAX_HEIGHT 8
#define QUAD_TREE_LEAF 32

#ifdef QUAD_TREE_PROFILE
constexpr bool quad_tree_profile_levels = true;
#else
constexpr bool quad_tree_profile_levels = false;
#endif

#endif