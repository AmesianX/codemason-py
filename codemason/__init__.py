"""
CodeMason - AI Coding Assistant CLI Tool
A Python reimplementation of Claude Code
"""

# The only place the version is written. pyproject.toml reads this attribute
# rather than carrying its own literal, the way the Rust client has Cargo.toml
# behind cli::VERSION -- two literals holding the same version is a version that
# can disagree with itself without anything noticing.
#
# Kept in lockstep with the Rust client: the two are meant to behave the same,
# so one number answering "are these the same build?" is worth more than each
# tree counting on its own. Bump both or neither.
__version__ = "3.0.0"
__author__ = "CodeMason Team"
