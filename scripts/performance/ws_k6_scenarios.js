import ws from 'k6/ws';
import { check, sleep } from 'k6';

export const options = {
  scenarios: {
    smoke: {
      executor: 'constant-vus',
      vus: Number(__ENV.VUS || 20),
      duration: __ENV.DURATION || '1m',
    },
  },
};

const WS_URL = __ENV.BACKEND_WS_URL || 'ws://localhost:18080/api/dashboard/ws/dashboard';
const TOKEN = __ENV.ACCESS_TOKEN || '';

export default function () {
  const url = TOKEN ? `${WS_URL}?access_token=${encodeURIComponent(TOKEN)}` : WS_URL
  const res = ws.connect(url, {}, function (socket) {
    socket.on('open', function () {
      socket.send(JSON.stringify({ type: 'ping', timestamp: new Date().toISOString() }));
    });
    socket.on('message', function (data) {
      // no-op
    });
    socket.on('close', function () {});
    socket.setTimeout(function () {
      socket.close();
    }, 5000);
  });
  check(res, { 'status is 101': (r) => r && r.status === 101 });
  sleep(1);
}
