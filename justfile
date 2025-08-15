# https://just.systems/man/en/

# REQUIRES

docker := require("docker")
find := require("find")
rm := require("rm")
uv := require("uv")

# SETTINGS

set dotenv-load := true

# VARIABLES

PACKAGE := "multimodal_ai_poc"
REPOSITORY := "multimodal-ai-poc"
SOURCES := "src"
TESTS := "tests"

# DEFAULTS

# display help information
default:
    @just --list

# IMPORTS

import 'tasks/check.just'
import 'tasks/clean.just'
import 'tasks/commit.just'
import 'tasks/doc.just'
import 'tasks/docker.just'
import 'tasks/format.just'
import 'tasks/install.just'
import 'tasks/package.just'
import 'tasks/project.just'
