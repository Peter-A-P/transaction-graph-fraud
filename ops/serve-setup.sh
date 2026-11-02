#!/usr/bin/env bash
# The serving benchmark's stores on a rented pod, after ops/pod-setup.sh. A pod is itself a
# container, so Neo4j runs natively (5.26 Community, the version the parity test ran against
# in Docker), with a password made here and never written anywhere but this pod's environment.
set -euo pipefail
cd "$(dirname "$0")/.."
apt-get update -qq && apt-get install -y -qq openjdk-21-jre-headless curl > /dev/null
if [ ! -d /opt/neo4j ]; then
  curl -fsSL https://dist.neo4j.org/neo4j-community-5.26.0-unix.tar.gz | tar -xz -C /opt
  mv /opt/neo4j-community-5.26.0 /opt/neo4j
fi
cat >> /opt/neo4j/conf/neo4j.conf <<CONF
server.memory.heap.initial_size=4g
server.memory.heap.max_size=8g
server.memory.pagecache.size=8g
CONF
pw=$(head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 24)
/opt/neo4j/bin/neo4j-admin dbms set-initial-password "$pw" > /dev/null
/opt/neo4j/bin/neo4j start > /dev/null
for i in $(seq 1 60); do curl -fs http://localhost:7474 > /dev/null && break; sleep 5; done
{ echo "NEO4J_URI=bolt://localhost:7687"; echo "NEO4J_USER=neo4j"; echo "NEO4J_PASSWORD=$pw"; } >> .env
echo "neo4j $(/opt/neo4j/bin/neo4j version)"
echo SERVESETUPDONE
