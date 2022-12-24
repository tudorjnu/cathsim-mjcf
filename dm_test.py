"""A global registry of constructors for manipulation environments."""


from dm_control.utils import containers

_ALL_CONSTRUCTORS = containers.TaggedTasks(allow_overriding_keys=False)

add = _ALL_CONSTRUCTORS.add
get_constructor = _ALL_CONSTRUCTORS.__getitem__
get_all_names = _ALL_CONSTRUCTORS.keys
get_tags = _ALL_CONSTRUCTORS.tags
get_names_by_tag = _ALL_CONSTRUCTORS.tagged

# This disables the check that prevents the same task constructor name from
# being added to the container more than once. This is done in order to allow
# individual task modules to be reloaded without also reloading `registry.py`
# first (e.g. when "hot-reloading" environments using IPython's `autoreload`
# extension).


def done_importing_tasks():
  _ALL_CONSTRUCTORS.allow_overriding_keys = True
