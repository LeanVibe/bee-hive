import {
  __commonJS
} from "./chunk-BUSYA2B4.js";

// node_modules/algoliasearch/dist/algoliasearch.umd.js
var require_algoliasearch_umd = __commonJS({
  "node_modules/algoliasearch/dist/algoliasearch.umd.js"(exports, module) {
    !function(e, t) {
      "object" == typeof exports && "undefined" != typeof module ? module.exports = t() : "function" == typeof define && define.amd ? define(t) : (e = e || self).algoliasearch = t();
    }(exports, function() {
      "use strict";
      function e(e2, t2, r2) {
        return t2 in e2 ? Object.defineProperty(e2, t2, { value: r2, enumerable: true, configurable: true, writable: true }) : e2[t2] = r2, e2;
      }
      function t(e2, t2) {
        var r2 = Object.keys(e2);
        if (Object.getOwnPropertySymbols) {
          var n2 = Object.getOwnPropertySymbols(e2);
          t2 && (n2 = n2.filter(function(t3) {
            return Object.getOwnPropertyDescriptor(e2, t3).enumerable;
          })), r2.push.apply(r2, n2);
        }
        return r2;
      }
      function r(r2) {
        for (var n2 = 1; n2 < arguments.length; n2++) {
          var a2 = null != arguments[n2] ? arguments[n2] : {};
          n2 % 2 ? t(Object(a2), true).forEach(function(t2) {
            e(r2, t2, a2[t2]);
          }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(r2, Object.getOwnPropertyDescriptors(a2)) : t(Object(a2)).forEach(function(e2) {
            Object.defineProperty(r2, e2, Object.getOwnPropertyDescriptor(a2, e2));
          });
        }
        return r2;
      }
      function n(e2, t2) {
        if (null == e2) return {};
        var r2, n2, a2 = function(e3, t3) {
          if (null == e3) return {};
          var r3, n3, a3 = {}, o3 = Object.keys(e3);
          for (n3 = 0; n3 < o3.length; n3++) r3 = o3[n3], t3.indexOf(r3) >= 0 || (a3[r3] = e3[r3]);
          return a3;
        }(e2, t2);
        if (Object.getOwnPropertySymbols) {
          var o2 = Object.getOwnPropertySymbols(e2);
          for (n2 = 0; n2 < o2.length; n2++) r2 = o2[n2], t2.indexOf(r2) >= 0 || Object.prototype.propertyIsEnumerable.call(e2, r2) && (a2[r2] = e2[r2]);
        }
        return a2;
      }
      function a(e2, t2) {
        return function(e3) {
          if (Array.isArray(e3)) return e3;
        }(e2) || function(e3, t3) {
          if (!(Symbol.iterator in Object(e3) || "[object Arguments]" === Object.prototype.toString.call(e3))) return;
          var r2 = [], n2 = true, a2 = false, o2 = void 0;
          try {
            for (var i2, u2 = e3[Symbol.iterator](); !(n2 = (i2 = u2.next()).done) && (r2.push(i2.value), !t3 || r2.length !== t3); n2 = true) ;
          } catch (e4) {
            a2 = true, o2 = e4;
          } finally {
            try {
              n2 || null == u2.return || u2.return();
            } finally {
              if (a2) throw o2;
            }
          }
          return r2;
        }(e2, t2) || function() {
          throw new TypeError("Invalid attempt to destructure non-iterable instance");
        }();
      }
      function o(e2) {
        return function(e3) {
          if (Array.isArray(e3)) {
            for (var t2 = 0, r2 = new Array(e3.length); t2 < e3.length; t2++) r2[t2] = e3[t2];
            return r2;
          }
        }(e2) || function(e3) {
          if (Symbol.iterator in Object(e3) || "[object Arguments]" === Object.prototype.toString.call(e3)) return Array.from(e3);
        }(e2) || function() {
          throw new TypeError("Invalid attempt to spread non-iterable instance");
        }();
      }
      function i(e2) {
        var t2, r2 = "algoliasearch-client-js-".concat(e2.key), n2 = function() {
          return void 0 === t2 && (t2 = e2.localStorage || window.localStorage), t2;
        }, o2 = function() {
          return JSON.parse(n2().getItem(r2) || "{}");
        }, i2 = function(e3) {
          n2().setItem(r2, JSON.stringify(e3));
        }, u2 = function() {
          var t3 = e2.timeToLive ? 1e3 * e2.timeToLive : null, r3 = o2(), n3 = Object.fromEntries(Object.entries(r3).filter(function(e3) {
            return void 0 !== a(e3, 2)[1].timestamp;
          }));
          if (i2(n3), t3) {
            var u3 = Object.fromEntries(Object.entries(n3).filter(function(e3) {
              var r4 = a(e3, 2)[1], n4 = (/* @__PURE__ */ new Date()).getTime();
              return !(r4.timestamp + t3 < n4);
            }));
            i2(u3);
          }
        };
        return { get: function(e3, t3) {
          var r3 = arguments.length > 2 && void 0 !== arguments[2] ? arguments[2] : { miss: function() {
            return Promise.resolve();
          } };
          return Promise.resolve().then(function() {
            u2();
            var t4 = JSON.stringify(e3);
            return o2()[t4];
          }).then(function(e4) {
            return Promise.all([e4 ? e4.value : t3(), void 0 !== e4]);
          }).then(function(e4) {
            var t4 = a(e4, 2), n3 = t4[0], o3 = t4[1];
            return Promise.all([n3, o3 || r3.miss(n3)]);
          }).then(function(e4) {
            return a(e4, 1)[0];
          });
        }, set: function(e3, t3) {
          return Promise.resolve().then(function() {
            var a2 = o2();
            return a2[JSON.stringify(e3)] = { timestamp: (/* @__PURE__ */ new Date()).getTime(), value: t3 }, n2().setItem(r2, JSON.stringify(a2)), t3;
          });
        }, delete: function(e3) {
          return Promise.resolve().then(function() {
            var t3 = o2();
            delete t3[JSON.stringify(e3)], n2().setItem(r2, JSON.stringify(t3));
          });
        }, clear: function() {
          return Promise.resolve().then(function() {
            n2().removeItem(r2);
          });
        } };
      }
      function u(e2) {
        var t2 = o(e2.caches), r2 = t2.shift();
        return void 0 === r2 ? { get: function(e3, t3) {
          var r3 = arguments.length > 2 && void 0 !== arguments[2] ? arguments[2] : { miss: function() {
            return Promise.resolve();
          } }, n2 = t3();
          return n2.then(function(e4) {
            return Promise.all([e4, r3.miss(e4)]);
          }).then(function(e4) {
            return a(e4, 1)[0];
          });
        }, set: function(e3, t3) {
          return Promise.resolve(t3);
        }, delete: function(e3) {
          return Promise.resolve();
        }, clear: function() {
          return Promise.resolve();
        } } : { get: function(e3, n2) {
          var a2 = arguments.length > 2 && void 0 !== arguments[2] ? arguments[2] : { miss: function() {
            return Promise.resolve();
          } };
          return r2.get(e3, n2, a2).catch(function() {
            return u({ caches: t2 }).get(e3, n2, a2);
          });
        }, set: function(e3, n2) {
          return r2.set(e3, n2).catch(function() {
            return u({ caches: t2 }).set(e3, n2);
          });
        }, delete: function(e3) {
          return r2.delete(e3).catch(function() {
            return u({ caches: t2 }).delete(e3);
          });
        }, clear: function() {
          return r2.clear().catch(function() {
            return u({ caches: t2 }).clear();
          });
        } };
      }
      function s() {
        var e2 = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : { serializable: true }, t2 = {};
        return { get: function(r2, n2) {
          var a2 = arguments.length > 2 && void 0 !== arguments[2] ? arguments[2] : { miss: function() {
            return Promise.resolve();
          } }, o2 = JSON.stringify(r2);
          if (o2 in t2) return Promise.resolve(e2.serializable ? JSON.parse(t2[o2]) : t2[o2]);
          var i2 = n2(), u2 = a2 && a2.miss || function() {
            return Promise.resolve();
          };
          return i2.then(function(e3) {
            return u2(e3);
          }).then(function() {
            return i2;
          });
        }, set: function(r2, n2) {
          return t2[JSON.stringify(r2)] = e2.serializable ? JSON.stringify(n2) : n2, Promise.resolve(n2);
        }, delete: function(e3) {
          return delete t2[JSON.stringify(e3)], Promise.resolve();
        }, clear: function() {
          return t2 = {}, Promise.resolve();
        } };
      }
      function c(e2, t2, r2) {
        var n2 = { "x-algolia-api-key": r2, "x-algolia-application-id": t2 };
        return { headers: function() {
          return e2 === m.WithinHeaders ? n2 : {};
        }, queryParameters: function() {
          return e2 === m.WithinQueryParameters ? n2 : {};
        } };
      }
      function f(e2) {
        var t2 = 0;
        return e2(function r2() {
          return t2++, new Promise(function(n2) {
            setTimeout(function() {
              n2(e2(r2));
            }, Math.min(100 * t2, 1e3));
          });
        });
      }
      function d(e2) {
        var t2 = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : function(e3, t3) {
          return Promise.resolve();
        };
        return Object.assign(e2, { wait: function(r2) {
          return d(e2.then(function(e3) {
            return Promise.all([t2(e3, r2), e3]);
          }).then(function(e3) {
            return e3[1];
          }));
        } });
      }
      function l(e2) {
        for (var t2 = e2.length - 1; t2 > 0; t2--) {
          var r2 = Math.floor(Math.random() * (t2 + 1)), n2 = e2[t2];
          e2[t2] = e2[r2], e2[r2] = n2;
        }
        return e2;
      }
      function h(e2, t2) {
        return t2 ? (Object.keys(t2).forEach(function(r2) {
          e2[r2] = t2[r2](e2);
        }), e2) : e2;
      }
      function p(e2) {
        for (var t2 = arguments.length, r2 = new Array(t2 > 1 ? t2 - 1 : 0), n2 = 1; n2 < t2; n2++) r2[n2 - 1] = arguments[n2];
        var a2 = 0;
        return e2.replace(/%s/g, function() {
          return encodeURIComponent(r2[a2++]);
        });
      }
      var m = { WithinQueryParameters: 0, WithinHeaders: 1 };
      function g(e2, t2) {
        var r2 = e2 || {}, n2 = r2.data || {};
        return Object.keys(r2).forEach(function(e3) {
          -1 === ["timeout", "headers", "queryParameters", "data", "cacheable"].indexOf(e3) && (n2[e3] = r2[e3]);
        }), { data: Object.entries(n2).length > 0 ? n2 : void 0, timeout: r2.timeout || t2, headers: r2.headers || {}, queryParameters: r2.queryParameters || {}, cacheable: r2.cacheable };
      }
      var y = { Read: 1, Write: 2, Any: 3 }, v = 1, b = 2, w = 3;
      function P(e2) {
        var t2 = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : v;
        return r(r({}, e2), {}, { status: t2, lastUpdate: Date.now() });
      }
      function O(e2) {
        return "string" == typeof e2 ? { protocol: "https", url: e2, accept: y.Any } : { protocol: e2.protocol || "https", url: e2.url, accept: e2.accept || y.Any };
      }
      var I = "DELETE", x = "GET", j = "POST", q = "PUT";
      function D(e2, t2) {
        return Promise.all(t2.map(function(t3) {
          return e2.get(t3, function() {
            return Promise.resolve(P(t3));
          });
        })).then(function(e3) {
          var r2 = e3.filter(function(e4) {
            return function(e5) {
              return e5.status === v || Date.now() - e5.lastUpdate > 12e4;
            }(e4);
          }), n2 = e3.filter(function(e4) {
            return function(e5) {
              return e5.status === w && Date.now() - e5.lastUpdate <= 12e4;
            }(e4);
          }), a2 = [].concat(o(r2), o(n2));
          return { getTimeout: function(e4, t3) {
            return (0 === n2.length && 0 === e4 ? 1 : n2.length + 3 + e4) * t3;
          }, statelessHosts: a2.length > 0 ? a2.map(function(e4) {
            return O(e4);
          }) : t2 };
        });
      }
      function T(e2, t2, n2, a2) {
        var i2 = [], u2 = function(e3, t3) {
          if (e3.method === x || void 0 === e3.data && void 0 === t3.data) return;
          var n3 = Array.isArray(e3.data) ? e3.data : r(r({}, e3.data), t3.data);
          return JSON.stringify(n3);
        }(n2, a2), s2 = function(e3, t3) {
          var n3 = r(r({}, e3.headers), t3.headers), a3 = {};
          return Object.keys(n3).forEach(function(e4) {
            var t4 = n3[e4];
            a3[e4.toLowerCase()] = t4;
          }), a3;
        }(e2, a2), c2 = n2.method, f2 = n2.method !== x ? {} : r(r({}, n2.data), a2.data), d2 = r(r(r({ "x-algolia-agent": e2.userAgent.value }, e2.queryParameters), f2), a2.queryParameters), l2 = 0, h2 = function t3(r2, o2) {
          var f3 = r2.pop();
          if (void 0 === f3) throw { name: "RetryError", message: "Unreachable hosts - your application id may be incorrect. If the error persists, please reach out to the Algolia Support team: https://alg.li/support .", transporterStackTrace: A(i2) };
          var h3 = { data: u2, headers: s2, method: c2, url: N(f3, n2.path, d2), connectTimeout: o2(l2, e2.timeouts.connect), responseTimeout: o2(l2, a2.timeout) }, p2 = function(e3) {
            var t4 = { request: h3, response: e3, host: f3, triesLeft: r2.length };
            return i2.push(t4), t4;
          }, m2 = { onSuccess: function(e3) {
            return function(e4) {
              try {
                return JSON.parse(e4.content);
              } catch (t4) {
                throw /* @__PURE__ */ function(e5, t5) {
                  return { name: "DeserializationError", message: e5, response: t5 };
                }(t4.message, e4);
              }
            }(e3);
          }, onRetry: function(n3) {
            var a3 = p2(n3);
            return n3.isTimedOut && l2++, Promise.all([e2.logger.info("Retryable failure", R(a3)), e2.hostsCache.set(f3, P(f3, n3.isTimedOut ? w : b))]).then(function() {
              return t3(r2, o2);
            });
          }, onFail: function(e3) {
            throw p2(e3), function(e4, t4) {
              var r3 = e4.content, n3 = e4.status, a3 = r3;
              try {
                a3 = JSON.parse(r3).message;
              } catch (e5) {
              }
              return /* @__PURE__ */ function(e5, t5, r4) {
                return { name: "ApiError", message: e5, status: t5, transporterStackTrace: r4 };
              }(a3, n3, t4);
            }(e3, A(i2));
          } };
          return e2.requester.send(h3).then(function(e3) {
            return function(e4, t4) {
              return function(e5) {
                var t5 = e5.status;
                return e5.isTimedOut || function(e6) {
                  var t6 = e6.isTimedOut, r3 = e6.status;
                  return !t6 && 0 == ~~r3;
                }(e5) || 2 != ~~(t5 / 100) && 4 != ~~(t5 / 100);
              }(e4) ? t4.onRetry(e4) : 2 == ~~(e4.status / 100) ? t4.onSuccess(e4) : t4.onFail(e4);
            }(e3, m2);
          });
        };
        return D(e2.hostsCache, t2).then(function(e3) {
          return h2(o(e3.statelessHosts).reverse(), e3.getTimeout);
        });
      }
      function k(e2) {
        var t2 = e2.hostsCache, r2 = e2.logger, n2 = e2.requester, o2 = e2.requestsCache, i2 = e2.responsesCache, u2 = e2.timeouts, s2 = e2.userAgent, c2 = e2.hosts, f2 = e2.queryParameters, d2 = { hostsCache: t2, logger: r2, requester: n2, requestsCache: o2, responsesCache: i2, timeouts: u2, userAgent: s2, headers: e2.headers, queryParameters: f2, hosts: c2.map(function(e3) {
          return O(e3);
        }), read: function(e3, t3) {
          var r3 = g(t3, d2.timeouts.read), n3 = function() {
            return T(d2, d2.hosts.filter(function(e4) {
              return 0 != (e4.accept & y.Read);
            }), e3, r3);
          };
          if (true !== (void 0 !== r3.cacheable ? r3.cacheable : e3.cacheable)) return n3();
          var o3 = { request: e3, mappedRequestOptions: r3, transporter: { queryParameters: d2.queryParameters, headers: d2.headers } };
          return d2.responsesCache.get(o3, function() {
            return d2.requestsCache.get(o3, function() {
              return d2.requestsCache.set(o3, n3()).then(function(e4) {
                return Promise.all([d2.requestsCache.delete(o3), e4]);
              }, function(e4) {
                return Promise.all([d2.requestsCache.delete(o3), Promise.reject(e4)]);
              }).then(function(e4) {
                var t4 = a(e4, 2);
                t4[0];
                return t4[1];
              });
            });
          }, { miss: function(e4) {
            return d2.responsesCache.set(o3, e4);
          } });
        }, write: function(e3, t3) {
          return T(d2, d2.hosts.filter(function(e4) {
            return 0 != (e4.accept & y.Write);
          }), e3, g(t3, d2.timeouts.write));
        } };
        return d2;
      }
      function S(e2) {
        var t2 = { value: "Algolia for JavaScript (".concat(e2, ")"), add: function(e3) {
          var r2 = "; ".concat(e3.segment).concat(void 0 !== e3.version ? " (".concat(e3.version, ")") : "");
          return -1 === t2.value.indexOf(r2) && (t2.value = "".concat(t2.value).concat(r2)), t2;
        } };
        return t2;
      }
      function N(e2, t2, r2) {
        var n2 = E(r2), a2 = "".concat(e2.protocol, "://").concat(e2.url, "/").concat("/" === t2.charAt(0) ? t2.substr(1) : t2);
        return n2.length && (a2 += "?".concat(n2)), a2;
      }
      function E(e2) {
        return Object.keys(e2).map(function(t2) {
          return p("%s=%s", t2, (r2 = e2[t2], "[object Object]" === Object.prototype.toString.call(r2) || "[object Array]" === Object.prototype.toString.call(r2) ? JSON.stringify(e2[t2]) : e2[t2]));
          var r2;
        }).join("&");
      }
      function A(e2) {
        return e2.map(function(e3) {
          return R(e3);
        });
      }
      function R(e2) {
        var t2 = e2.request.headers["x-algolia-api-key"] ? { "x-algolia-api-key": "*****" } : {};
        return r(r({}, e2), {}, { request: r(r({}, e2.request), {}, { headers: r(r({}, e2.request.headers), t2) }) });
      }
      var C = function(e2) {
        return function(t2, r2) {
          return e2.transporter.write({ method: j, path: "2/abtests", data: t2 }, r2);
        };
      }, U = function(e2) {
        return function(t2, r2) {
          return e2.transporter.write({ method: I, path: p("2/abtests/%s", t2) }, r2);
        };
      }, z = function(e2) {
        return function(t2, r2) {
          return e2.transporter.read({ method: x, path: p("2/abtests/%s", t2) }, r2);
        };
      }, J = function(e2) {
        return function(t2) {
          return e2.transporter.read({ method: x, path: "2/abtests" }, t2);
        };
      }, F = function(e2) {
        return function(t2, r2) {
          return e2.transporter.write({ method: j, path: p("2/abtests/%s/stop", t2) }, r2);
        };
      }, W = function(e2) {
        return function(t2) {
          return e2.transporter.read({ method: x, path: "1/strategies/personalization" }, t2);
        };
      }, H = function(e2) {
        return function(t2, r2) {
          return e2.transporter.write({ method: j, path: "1/strategies/personalization", data: t2 }, r2);
        };
      };
      function K(e2) {
        return function t2(r2) {
          return e2.request(r2).then(function(n2) {
            if (void 0 !== e2.batch && e2.batch(n2.hits), !e2.shouldStop(n2)) return n2.cursor ? t2({ cursor: n2.cursor }) : t2({ page: (r2.page || 0) + 1 });
          });
        }({});
      }
      var M = function(e2) {
        return function(t2, a2) {
          var o2 = a2 || {}, i2 = o2.queryParameters, u2 = n(o2, ["queryParameters"]), s2 = r({ acl: t2 }, void 0 !== i2 ? { queryParameters: i2 } : {});
          return d(e2.transporter.write({ method: j, path: "1/keys", data: s2 }, u2), function(t3, r2) {
            return f(function(n2) {
              return ee(e2)(t3.key, r2).catch(function(e3) {
                if (404 !== e3.status) throw e3;
                return n2();
              });
            });
          });
        };
      }, B = function(e2) {
        return function(t2, r2, n2) {
          var a2 = g(n2);
          return a2.queryParameters["X-Algolia-User-ID"] = t2, e2.transporter.write({ method: j, path: "1/clusters/mapping", data: { cluster: r2 } }, a2);
        };
      }, G = function(e2) {
        return function(t2, r2, n2) {
          return e2.transporter.write({ method: j, path: "1/clusters/mapping/batch", data: { users: t2, cluster: r2 } }, n2);
        };
      }, L = function(e2) {
        return function(t2, r2) {
          return d(e2.transporter.write({ method: j, path: p("/1/dictionaries/%s/batch", t2), data: { clearExistingDictionaryEntries: true, requests: { action: "addEntry", body: [] } } }, r2), function(t3, r3) {
            return je(e2)(t3.taskID, r3);
          });
        };
      }, Q = function(e2) {
        return function(t2, r2, n2) {
          return d(e2.transporter.write({ method: j, path: p("1/indexes/%s/operation", t2), data: { operation: "copy", destination: r2 } }, n2), function(r3, n3) {
            return ue(e2)(t2, { methods: { waitTask: dt } }).waitTask(r3.taskID, n3);
          });
        };
      }, V = function(e2) {
        return function(t2, n2, a2) {
          return Q(e2)(t2, n2, r(r({}, a2), {}, { scope: [ht.Rules] }));
        };
      }, _ = function(e2) {
        return function(t2, n2, a2) {
          return Q(e2)(t2, n2, r(r({}, a2), {}, { scope: [ht.Settings] }));
        };
      }, X = function(e2) {
        return function(t2, n2, a2) {
          return Q(e2)(t2, n2, r(r({}, a2), {}, { scope: [ht.Synonyms] }));
        };
      }, Y = function(e2) {
        return function(t2, r2) {
          return t2.method === x ? e2.transporter.read(t2, r2) : e2.transporter.write(t2, r2);
        };
      }, Z = function(e2) {
        return function(t2, r2) {
          return d(e2.transporter.write({ method: I, path: p("1/keys/%s", t2) }, r2), function(r3, n2) {
            return f(function(r4) {
              return ee(e2)(t2, n2).then(r4).catch(function(e3) {
                if (404 !== e3.status) throw e3;
              });
            });
          });
        };
      }, $ = function(e2) {
        return function(t2, r2, n2) {
          var a2 = r2.map(function(e3) {
            return { action: "deleteEntry", body: { objectID: e3 } };
          });
          return d(e2.transporter.write({ method: j, path: p("/1/dictionaries/%s/batch", t2), data: { clearExistingDictionaryEntries: false, requests: a2 } }, n2), function(t3, r3) {
            return je(e2)(t3.taskID, r3);
          });
        };
      }, ee = function(e2) {
        return function(t2, r2) {
          return e2.transporter.read({ method: x, path: p("1/keys/%s", t2) }, r2);
        };
      }, te = function(e2) {
        return function(t2, r2) {
          return e2.transporter.read({ method: x, path: p("1/task/%s", t2.toString()) }, r2);
        };
      }, re = function(e2) {
        return function(t2) {
          return e2.transporter.read({ method: x, path: "/1/dictionaries/*/settings" }, t2);
        };
      }, ne = function(e2) {
        return function(t2) {
          return e2.transporter.read({ method: x, path: "1/logs" }, t2);
        };
      }, ae = function(e2) {
        return function(t2) {
          return e2.transporter.read({ method: x, path: "1/clusters/mapping/top" }, t2);
        };
      }, oe = function(e2) {
        return function(t2, r2) {
          return e2.transporter.read({ method: x, path: p("1/clusters/mapping/%s", t2) }, r2);
        };
      }, ie = function(e2) {
        return function(t2) {
          var r2 = t2 || {}, a2 = r2.retrieveMappings, o2 = n(r2, ["retrieveMappings"]);
          return true === a2 && (o2.getClusters = true), e2.transporter.read({ method: x, path: "1/clusters/mapping/pending" }, o2);
        };
      }, ue = function(e2) {
        return function(t2) {
          var r2 = arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {}, n2 = { transporter: e2.transporter, appId: e2.appId, indexName: t2 };
          return h(n2, r2.methods);
        };
      }, se = function(e2) {
        return function(t2) {
          return e2.transporter.read({ method: x, path: "1/keys" }, t2);
        };
      }, ce = function(e2) {
        return function(t2) {
          return e2.transporter.read({ method: x, path: "1/clusters" }, t2);
        };
      }, fe = function(e2) {
        return function(t2) {
          return e2.transporter.read({ method: x, path: "1/indexes" }, t2);
        };
      }, de = function(e2) {
        return function(t2) {
          return e2.transporter.read({ method: x, path: "1/clusters/mapping" }, t2);
        };
      }, le = function(e2) {
        return function(t2, r2, n2) {
          return d(e2.transporter.write({ method: j, path: p("1/indexes/%s/operation", t2), data: { operation: "move", destination: r2 } }, n2), function(r3, n3) {
            return ue(e2)(t2, { methods: { waitTask: dt } }).waitTask(r3.taskID, n3);
          });
        };
      }, he = function(e2) {
        return function(t2, r2) {
          return d(e2.transporter.write({ method: j, path: "1/indexes/*/batch", data: { requests: t2 } }, r2), function(t3, r3) {
            return Promise.all(Object.keys(t3.taskID).map(function(n2) {
              return ue(e2)(n2, { methods: { waitTask: dt } }).waitTask(t3.taskID[n2], r3);
            }));
          });
        };
      }, pe = function(e2) {
        return function(t2, r2) {
          return e2.transporter.read({ method: j, path: "1/indexes/*/objects", data: { requests: t2 } }, r2);
        };
      }, me = function(e2) {
        return function(t2, n2) {
          var a2 = t2.map(function(e3) {
            return r(r({}, e3), {}, { params: E(e3.params || {}) });
          });
          return e2.transporter.read({ method: j, path: "1/indexes/*/queries", data: { requests: a2 }, cacheable: true }, n2);
        };
      }, ge = function(e2) {
        return function(t2, a2) {
          return Promise.all(t2.map(function(t3) {
            var o2 = t3.params, i2 = o2.facetName, u2 = o2.facetQuery, s2 = n(o2, ["facetName", "facetQuery"]);
            return ue(e2)(t3.indexName, { methods: { searchForFacetValues: ut } }).searchForFacetValues(i2, u2, r(r({}, a2), s2));
          }));
        };
      }, ye = function(e2) {
        return function(t2, r2) {
          var n2 = g(r2);
          return n2.queryParameters["X-Algolia-User-ID"] = t2, e2.transporter.write({ method: I, path: "1/clusters/mapping" }, n2);
        };
      }, ve = function(e2) {
        return function(t2, r2, n2) {
          var a2 = r2.map(function(e3) {
            return { action: "addEntry", body: e3 };
          });
          return d(e2.transporter.write({ method: j, path: p("/1/dictionaries/%s/batch", t2), data: { clearExistingDictionaryEntries: true, requests: a2 } }, n2), function(t3, r3) {
            return je(e2)(t3.taskID, r3);
          });
        };
      }, be = function(e2) {
        return function(t2, r2) {
          return d(e2.transporter.write({ method: j, path: p("1/keys/%s/restore", t2) }, r2), function(r3, n2) {
            return f(function(r4) {
              return ee(e2)(t2, n2).catch(function(e3) {
                if (404 !== e3.status) throw e3;
                return r4();
              });
            });
          });
        };
      }, we = function(e2) {
        return function(t2, r2, n2) {
          var a2 = r2.map(function(e3) {
            return { action: "addEntry", body: e3 };
          });
          return d(e2.transporter.write({ method: j, path: p("/1/dictionaries/%s/batch", t2), data: { clearExistingDictionaryEntries: false, requests: a2 } }, n2), function(t3, r3) {
            return je(e2)(t3.taskID, r3);
          });
        };
      }, Pe = function(e2) {
        return function(t2, r2, n2) {
          return e2.transporter.read({ method: j, path: p("/1/dictionaries/%s/search", t2), data: { query: r2 }, cacheable: true }, n2);
        };
      }, Oe = function(e2) {
        return function(t2, r2) {
          return e2.transporter.read({ method: j, path: "1/clusters/mapping/search", data: { query: t2 } }, r2);
        };
      }, Ie = function(e2) {
        return function(t2, r2) {
          return d(e2.transporter.write({ method: q, path: "/1/dictionaries/*/settings", data: t2 }, r2), function(t3, r3) {
            return je(e2)(t3.taskID, r3);
          });
        };
      }, xe = function(e2) {
        return function(t2, r2) {
          var a2 = Object.assign({}, r2), o2 = r2 || {}, i2 = o2.queryParameters, u2 = n(o2, ["queryParameters"]), s2 = i2 ? { queryParameters: i2 } : {}, c2 = ["acl", "indexes", "referers", "restrictSources", "queryParameters", "description", "maxQueriesPerIPPerHour", "maxHitsPerQuery"];
          return d(e2.transporter.write({ method: q, path: p("1/keys/%s", t2), data: s2 }, u2), function(r3, n2) {
            return f(function(r4) {
              return ee(e2)(t2, n2).then(function(e3) {
                return function(e4) {
                  return Object.keys(a2).filter(function(e5) {
                    return -1 !== c2.indexOf(e5);
                  }).every(function(t3) {
                    if (Array.isArray(e4[t3]) && Array.isArray(a2[t3])) {
                      var r5 = e4[t3];
                      return r5.length === a2[t3].length && r5.every(function(e5, r6) {
                        return e5 === a2[t3][r6];
                      });
                    }
                    return e4[t3] === a2[t3];
                  });
                }(e3) ? Promise.resolve() : r4();
              });
            });
          });
        };
      }, je = function(e2) {
        return function(t2, r2) {
          return f(function(n2) {
            return te(e2)(t2, r2).then(function(e3) {
              return "published" !== e3.status ? n2() : void 0;
            });
          });
        };
      }, qe = function(e2) {
        return function(t2, r2) {
          return d(e2.transporter.write({ method: j, path: p("1/indexes/%s/batch", e2.indexName), data: { requests: t2 } }, r2), function(t3, r3) {
            return dt(e2)(t3.taskID, r3);
          });
        };
      }, De = function(e2) {
        return function(t2) {
          return K(r(r({ shouldStop: function(e3) {
            return void 0 === e3.cursor;
          } }, t2), {}, { request: function(r2) {
            return e2.transporter.read({ method: j, path: p("1/indexes/%s/browse", e2.indexName), data: r2 }, t2);
          } }));
        };
      }, Te = function(e2) {
        return function(t2) {
          var n2 = r({ hitsPerPage: 1e3 }, t2);
          return K(r(r({ shouldStop: function(e3) {
            return e3.hits.length < n2.hitsPerPage;
          } }, n2), {}, { request: function(t3) {
            return st(e2)("", r(r({}, n2), t3)).then(function(e3) {
              return r(r({}, e3), {}, { hits: e3.hits.map(function(e4) {
                return delete e4._highlightResult, e4;
              }) });
            });
          } }));
        };
      }, ke = function(e2) {
        return function(t2) {
          var n2 = r({ hitsPerPage: 1e3 }, t2);
          return K(r(r({ shouldStop: function(e3) {
            return e3.hits.length < n2.hitsPerPage;
          } }, n2), {}, { request: function(t3) {
            return ct(e2)("", r(r({}, n2), t3)).then(function(e3) {
              return r(r({}, e3), {}, { hits: e3.hits.map(function(e4) {
                return delete e4._highlightResult, e4;
              }) });
            });
          } }));
        };
      }, Se = function(e2) {
        return function(t2, r2, a2) {
          var o2 = a2 || {}, i2 = o2.batchSize, u2 = n(o2, ["batchSize"]), s2 = { taskIDs: [], objectIDs: [] };
          return d(function n2() {
            var a3, o3 = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : 0, c2 = [];
            for (a3 = o3; a3 < t2.length && (c2.push(t2[a3]), c2.length !== (i2 || 1e3)); a3++) ;
            return 0 === c2.length ? Promise.resolve(s2) : qe(e2)(c2.map(function(e3) {
              return { action: r2, body: e3 };
            }), u2).then(function(e3) {
              return s2.objectIDs = s2.objectIDs.concat(e3.objectIDs), s2.taskIDs.push(e3.taskID), a3++, n2(a3);
            });
          }(), function(t3, r3) {
            return Promise.all(t3.taskIDs.map(function(t4) {
              return dt(e2)(t4, r3);
            }));
          });
        };
      }, Ne = function(e2) {
        return function(t2) {
          return d(e2.transporter.write({ method: j, path: p("1/indexes/%s/clear", e2.indexName) }, t2), function(t3, r2) {
            return dt(e2)(t3.taskID, r2);
          });
        };
      }, Ee = function(e2) {
        return function(t2) {
          var r2 = t2 || {}, a2 = r2.forwardToReplicas, o2 = g(n(r2, ["forwardToReplicas"]));
          return a2 && (o2.queryParameters.forwardToReplicas = 1), d(e2.transporter.write({ method: j, path: p("1/indexes/%s/rules/clear", e2.indexName) }, o2), function(t3, r3) {
            return dt(e2)(t3.taskID, r3);
          });
        };
      }, Ae = function(e2) {
        return function(t2) {
          var r2 = t2 || {}, a2 = r2.forwardToReplicas, o2 = g(n(r2, ["forwardToReplicas"]));
          return a2 && (o2.queryParameters.forwardToReplicas = 1), d(e2.transporter.write({ method: j, path: p("1/indexes/%s/synonyms/clear", e2.indexName) }, o2), function(t3, r3) {
            return dt(e2)(t3.taskID, r3);
          });
        };
      }, Re = function(e2) {
        return function(t2, r2) {
          return d(e2.transporter.write({ method: j, path: p("1/indexes/%s/deleteByQuery", e2.indexName), data: t2 }, r2), function(t3, r3) {
            return dt(e2)(t3.taskID, r3);
          });
        };
      }, Ce = function(e2) {
        return function(t2) {
          return d(e2.transporter.write({ method: I, path: p("1/indexes/%s", e2.indexName) }, t2), function(t3, r2) {
            return dt(e2)(t3.taskID, r2);
          });
        };
      }, Ue = function(e2) {
        return function(t2, r2) {
          return d(ze(e2)([t2], r2).then(function(e3) {
            return { taskID: e3.taskIDs[0] };
          }), function(t3, r3) {
            return dt(e2)(t3.taskID, r3);
          });
        };
      }, ze = function(e2) {
        return function(t2, r2) {
          var n2 = t2.map(function(e3) {
            return { objectID: e3 };
          });
          return Se(e2)(n2, lt.DeleteObject, r2);
        };
      }, Je = function(e2) {
        return function(t2, r2) {
          var a2 = r2 || {}, o2 = a2.forwardToReplicas, i2 = g(n(a2, ["forwardToReplicas"]));
          return o2 && (i2.queryParameters.forwardToReplicas = 1), d(e2.transporter.write({ method: I, path: p("1/indexes/%s/rules/%s", e2.indexName, t2) }, i2), function(t3, r3) {
            return dt(e2)(t3.taskID, r3);
          });
        };
      }, Fe = function(e2) {
        return function(t2, r2) {
          var a2 = r2 || {}, o2 = a2.forwardToReplicas, i2 = g(n(a2, ["forwardToReplicas"]));
          return o2 && (i2.queryParameters.forwardToReplicas = 1), d(e2.transporter.write({ method: I, path: p("1/indexes/%s/synonyms/%s", e2.indexName, t2) }, i2), function(t3, r3) {
            return dt(e2)(t3.taskID, r3);
          });
        };
      }, We = function(e2) {
        return function(t2) {
          return Qe(e2)(t2).then(function() {
            return true;
          }).catch(function(e3) {
            if (404 !== e3.status) throw e3;
            return false;
          });
        };
      }, He = function(e2) {
        return function(t2, r2, n2) {
          return e2.transporter.read({ method: j, path: p("1/answers/%s/prediction", e2.indexName), data: { query: t2, queryLanguages: r2 }, cacheable: true }, n2);
        };
      }, Ke = function(e2) {
        return function(t2, o2) {
          var i2 = o2 || {}, u2 = i2.query, s2 = i2.paginate, c2 = n(i2, ["query", "paginate"]), f2 = 0;
          return function n2() {
            return it(e2)(u2 || "", r(r({}, c2), {}, { page: f2 })).then(function(e3) {
              for (var r2 = 0, o3 = Object.entries(e3.hits); r2 < o3.length; r2++) {
                var i3 = a(o3[r2], 2), u3 = i3[0], c3 = i3[1];
                if (t2(c3)) return { object: c3, position: parseInt(u3, 10), page: f2 };
              }
              if (f2++, false === s2 || f2 >= e3.nbPages) throw { name: "ObjectNotFoundError", message: "Object not found." };
              return n2();
            });
          }();
        };
      }, Me = function(e2) {
        return function(t2, r2) {
          return e2.transporter.read({ method: x, path: p("1/indexes/%s/%s", e2.indexName, t2) }, r2);
        };
      }, Be = function() {
        return function(e2, t2) {
          for (var r2 = 0, n2 = Object.entries(e2.hits); r2 < n2.length; r2++) {
            var o2 = a(n2[r2], 2), i2 = o2[0];
            if (o2[1].objectID === t2) return parseInt(i2, 10);
          }
          return -1;
        };
      }, Ge = function(e2) {
        return function(t2, a2) {
          var o2 = a2 || {}, i2 = o2.attributesToRetrieve, u2 = n(o2, ["attributesToRetrieve"]), s2 = t2.map(function(t3) {
            return r({ indexName: e2.indexName, objectID: t3 }, i2 ? { attributesToRetrieve: i2 } : {});
          });
          return e2.transporter.read({ method: j, path: "1/indexes/*/objects", data: { requests: s2 } }, u2);
        };
      }, Le = function(e2) {
        return function(t2, r2) {
          return e2.transporter.read({ method: x, path: p("1/indexes/%s/rules/%s", e2.indexName, t2) }, r2);
        };
      }, Qe = function(e2) {
        return function(t2) {
          return e2.transporter.read({ method: x, path: p("1/indexes/%s/settings", e2.indexName), data: { getVersion: 2 } }, t2);
        };
      }, Ve = function(e2) {
        return function(t2, r2) {
          return e2.transporter.read({ method: x, path: p("1/indexes/%s/synonyms/%s", e2.indexName, t2) }, r2);
        };
      }, _e = function(e2) {
        return function(t2, r2) {
          return d(Xe(e2)([t2], r2).then(function(e3) {
            return { objectID: e3.objectIDs[0], taskID: e3.taskIDs[0] };
          }), function(t3, r3) {
            return dt(e2)(t3.taskID, r3);
          });
        };
      }, Xe = function(e2) {
        return function(t2, r2) {
          var a2 = r2 || {}, o2 = a2.createIfNotExists, i2 = n(a2, ["createIfNotExists"]), u2 = o2 ? lt.PartialUpdateObject : lt.PartialUpdateObjectNoCreate;
          return Se(e2)(t2, u2, i2);
        };
      }, Ye = function(e2) {
        return function(t2, i2) {
          var u2 = i2 || {}, s2 = u2.safe, c2 = u2.autoGenerateObjectIDIfNotExist, f2 = u2.batchSize, l2 = n(u2, ["safe", "autoGenerateObjectIDIfNotExist", "batchSize"]), h2 = function(t3, r2, n2, a2) {
            return d(e2.transporter.write({ method: j, path: p("1/indexes/%s/operation", t3), data: { operation: n2, destination: r2 } }, a2), function(t4, r3) {
              return dt(e2)(t4.taskID, r3);
            });
          }, m2 = Math.random().toString(36).substring(7), g2 = "".concat(e2.indexName, "_tmp_").concat(m2), y2 = tt({ appId: e2.appId, transporter: e2.transporter, indexName: g2 }), v2 = [], b2 = h2(e2.indexName, g2, "copy", r(r({}, l2), {}, { scope: ["settings", "synonyms", "rules"] }));
          return v2.push(b2), d((s2 ? b2.wait(l2) : b2).then(function() {
            var e3 = y2(t2, r(r({}, l2), {}, { autoGenerateObjectIDIfNotExist: c2, batchSize: f2 }));
            return v2.push(e3), s2 ? e3.wait(l2) : e3;
          }).then(function() {
            var t3 = h2(g2, e2.indexName, "move", l2);
            return v2.push(t3), s2 ? t3.wait(l2) : t3;
          }).then(function() {
            return Promise.all(v2);
          }).then(function(e3) {
            var t3 = a(e3, 3), r2 = t3[0], n2 = t3[1], i3 = t3[2];
            return { objectIDs: n2.objectIDs, taskIDs: [r2.taskID].concat(o(n2.taskIDs), [i3.taskID]) };
          }), function(e3, t3) {
            return Promise.all(v2.map(function(e4) {
              return e4.wait(t3);
            }));
          });
        };
      }, Ze = function(e2) {
        return function(t2, n2) {
          return nt(e2)(t2, r(r({}, n2), {}, { clearExistingRules: true }));
        };
      }, $e = function(e2) {
        return function(t2, n2) {
          return ot(e2)(t2, r(r({}, n2), {}, { clearExistingSynonyms: true }));
        };
      }, et = function(e2) {
        return function(t2, r2) {
          return d(tt(e2)([t2], r2).then(function(e3) {
            return { objectID: e3.objectIDs[0], taskID: e3.taskIDs[0] };
          }), function(t3, r3) {
            return dt(e2)(t3.taskID, r3);
          });
        };
      }, tt = function(e2) {
        return function(t2, r2) {
          var a2 = r2 || {}, o2 = a2.autoGenerateObjectIDIfNotExist, i2 = n(a2, ["autoGenerateObjectIDIfNotExist"]), u2 = o2 ? lt.AddObject : lt.UpdateObject;
          if (u2 === lt.UpdateObject) {
            var s2 = true, c2 = false, f2 = void 0;
            try {
              for (var l2, h2 = t2[Symbol.iterator](); !(s2 = (l2 = h2.next()).done); s2 = true) {
                if (void 0 === l2.value.objectID) return d(Promise.reject({ name: "MissingObjectIDError", message: "All objects must have an unique objectID (like a primary key) to be valid. Algolia is also able to generate objectIDs automatically but *it's not recommended*. To do it, use the `{'autoGenerateObjectIDIfNotExist': true}` option." }));
              }
            } catch (e3) {
              c2 = true, f2 = e3;
            } finally {
              try {
                s2 || null == h2.return || h2.return();
              } finally {
                if (c2) throw f2;
              }
            }
          }
          return Se(e2)(t2, u2, i2);
        };
      }, rt = function(e2) {
        return function(t2, r2) {
          return nt(e2)([t2], r2);
        };
      }, nt = function(e2) {
        return function(t2, r2) {
          var a2 = r2 || {}, o2 = a2.forwardToReplicas, i2 = a2.clearExistingRules, u2 = g(n(a2, ["forwardToReplicas", "clearExistingRules"]));
          return o2 && (u2.queryParameters.forwardToReplicas = 1), i2 && (u2.queryParameters.clearExistingRules = 1), d(e2.transporter.write({ method: j, path: p("1/indexes/%s/rules/batch", e2.indexName), data: t2 }, u2), function(t3, r3) {
            return dt(e2)(t3.taskID, r3);
          });
        };
      }, at = function(e2) {
        return function(t2, r2) {
          return ot(e2)([t2], r2);
        };
      }, ot = function(e2) {
        return function(t2, r2) {
          var a2 = r2 || {}, o2 = a2.forwardToReplicas, i2 = a2.clearExistingSynonyms, u2 = a2.replaceExistingSynonyms, s2 = g(n(a2, ["forwardToReplicas", "clearExistingSynonyms", "replaceExistingSynonyms"]));
          return o2 && (s2.queryParameters.forwardToReplicas = 1), (u2 || i2) && (s2.queryParameters.replaceExistingSynonyms = 1), d(e2.transporter.write({ method: j, path: p("1/indexes/%s/synonyms/batch", e2.indexName), data: t2 }, s2), function(t3, r3) {
            return dt(e2)(t3.taskID, r3);
          });
        };
      }, it = function(e2) {
        return function(t2, r2) {
          return e2.transporter.read({ method: j, path: p("1/indexes/%s/query", e2.indexName), data: { query: t2 }, cacheable: true }, r2);
        };
      }, ut = function(e2) {
        return function(t2, r2, n2) {
          return e2.transporter.read({ method: j, path: p("1/indexes/%s/facets/%s/query", e2.indexName, t2), data: { facetQuery: r2 }, cacheable: true }, n2);
        };
      }, st = function(e2) {
        return function(t2, r2) {
          return e2.transporter.read({ method: j, path: p("1/indexes/%s/rules/search", e2.indexName), data: { query: t2 } }, r2);
        };
      }, ct = function(e2) {
        return function(t2, r2) {
          return e2.transporter.read({ method: j, path: p("1/indexes/%s/synonyms/search", e2.indexName), data: { query: t2 } }, r2);
        };
      }, ft = function(e2) {
        return function(t2, r2) {
          var a2 = r2 || {}, o2 = a2.forwardToReplicas, i2 = g(n(a2, ["forwardToReplicas"]));
          return o2 && (i2.queryParameters.forwardToReplicas = 1), d(e2.transporter.write({ method: q, path: p("1/indexes/%s/settings", e2.indexName), data: t2 }, i2), function(t3, r3) {
            return dt(e2)(t3.taskID, r3);
          });
        };
      }, dt = function(e2) {
        return function(t2, r2) {
          return f(function(n2) {
            return (/* @__PURE__ */ function(e3) {
              return function(t3, r3) {
                return e3.transporter.read({ method: x, path: p("1/indexes/%s/task/%s", e3.indexName, t3.toString()) }, r3);
              };
            }(e2))(t2, r2).then(function(e3) {
              return "published" !== e3.status ? n2() : void 0;
            });
          });
        };
      }, lt = { AddObject: "addObject", UpdateObject: "updateObject", PartialUpdateObject: "partialUpdateObject", PartialUpdateObjectNoCreate: "partialUpdateObjectNoCreate", DeleteObject: "deleteObject", DeleteIndex: "delete", ClearIndex: "clear" }, ht = { Settings: "settings", Synonyms: "synonyms", Rules: "rules" }, pt = 1, mt = 2, gt = 3;
      var yt = function(e2) {
        return function(t2, n2) {
          var a2 = t2.map(function(e3) {
            return r(r({}, e3), {}, { threshold: e3.threshold || 0 });
          });
          return e2.transporter.read({ method: j, path: "1/indexes/*/recommendations", data: { requests: a2 }, cacheable: true }, n2);
        };
      }, vt = function(e2) {
        return function(t2, n2) {
          return yt(e2)(t2.map(function(e3) {
            return r(r({}, e3), {}, { fallbackParameters: {}, model: "bought-together" });
          }), n2);
        };
      }, bt = function(e2) {
        return function(t2, n2) {
          return yt(e2)(t2.map(function(e3) {
            return r(r({}, e3), {}, { model: "related-products" });
          }), n2);
        };
      }, wt = function(e2) {
        return function(t2, n2) {
          var a2 = t2.map(function(e3) {
            return r(r({}, e3), {}, { model: "trending-facets", threshold: e3.threshold || 0 });
          });
          return e2.transporter.read({ method: j, path: "1/indexes/*/recommendations", data: { requests: a2 }, cacheable: true }, n2);
        };
      }, Pt = function(e2) {
        return function(t2, n2) {
          var a2 = t2.map(function(e3) {
            return r(r({}, e3), {}, { model: "trending-items", threshold: e3.threshold || 0 });
          });
          return e2.transporter.read({ method: j, path: "1/indexes/*/recommendations", data: { requests: a2 }, cacheable: true }, n2);
        };
      }, Ot = function(e2) {
        return function(t2, n2) {
          return yt(e2)(t2.map(function(e3) {
            return r(r({}, e3), {}, { model: "looking-similar" });
          }), n2);
        };
      }, It = function(e2) {
        return function(t2, n2) {
          var a2 = t2.map(function(e3) {
            return r(r({}, e3), {}, { model: "recommended-for-you", threshold: e3.threshold || 0 });
          });
          return e2.transporter.read({ method: j, path: "1/indexes/*/recommendations", data: { requests: a2 }, cacheable: true }, n2);
        };
      };
      function xt(e2, t2) {
        return function(r2, a2) {
          if (!t2) throw qt("`options.transformation.region` must be provided at client instantiation before calling this method.");
          var o2 = a2 || {}, i2 = o2.autoGenerateObjectIDIfNotExist, u2 = o2.watch, s2 = n(o2, ["autoGenerateObjectIDIfNotExist", "watch"]), c2 = i2 ? lt.AddObject : lt.UpdateObject;
          return t2.push({ indexName: e2, pushTaskPayload: { action: c2, records: r2 }, watch: u2 }, s2);
        };
      }
      function jt(e2, t2) {
        return function(r2, a2) {
          if (!t2) throw qt("`options.transformation.region` must be provided at client instantiation before calling this method.");
          var o2 = a2 || {}, i2 = o2.createIfNotExists, u2 = o2.watch, s2 = n(o2, ["createIfNotExists", "watch"]), c2 = i2 ? lt.PartialUpdateObject : lt.PartialUpdateObjectNoCreate;
          return t2.push({ indexName: e2, pushTaskPayload: { action: c2, records: r2 }, watch: u2 }, s2);
        };
      }
      function qt(e2) {
        return { name: "TransformationConfigurationError", message: e2 };
      }
      function Dt(e2, t2, n2) {
        var a2, o2, f2 = { appId: e2, apiKey: t2, timeouts: { connect: 1, read: 2, write: 30 }, requester: { send: function(e3) {
          return new Promise(function(t3) {
            var r2 = new XMLHttpRequest();
            r2.open(e3.method, e3.url, true), Object.keys(e3.headers).forEach(function(t4) {
              return r2.setRequestHeader(t4, e3.headers[t4]);
            });
            var n3, a3 = function(e4, n4) {
              return setTimeout(function() {
                r2.abort(), t3({ status: 0, content: n4, isTimedOut: true });
              }, 1e3 * e4);
            }, o3 = a3(e3.connectTimeout, "Connection timeout");
            r2.onreadystatechange = function() {
              r2.readyState > r2.OPENED && void 0 === n3 && (clearTimeout(o3), n3 = a3(e3.responseTimeout, "Socket timeout"));
            }, r2.onerror = function() {
              0 === r2.status && (clearTimeout(o3), clearTimeout(n3), t3({ content: r2.responseText || "Network request failed", status: r2.status, isTimedOut: false }));
            }, r2.onload = function() {
              clearTimeout(o3), clearTimeout(n3), t3({ content: r2.responseText, status: r2.status, isTimedOut: false });
            }, r2.send(e3.data);
          });
        } }, logger: (a2 = gt, { debug: function(e3, t3) {
          return pt >= a2 && console.debug(e3, t3), Promise.resolve();
        }, info: function(e3, t3) {
          return mt >= a2 && console.info(e3, t3), Promise.resolve();
        }, error: function(e3, t3) {
          return console.error(e3, t3), Promise.resolve();
        } }), responsesCache: s(), requestsCache: s({ serializable: false }), hostsCache: u({ caches: [i({ key: "".concat("4.25.2", "-").concat(e2) }), s()] }), userAgent: S("4.25.2").add({ segment: "Browser" }) }, d2 = r(r({}, f2), n2), g2 = function() {
          return function(e3) {
            return function(e4) {
              var t3 = e4.region || "us", n3 = c(m.WithinHeaders, e4.appId, e4.apiKey), a3 = k(r(r({ hosts: [{ url: "personalization.".concat(t3, ".algolia.com") }] }, e4), {}, { headers: r(r(r({}, n3.headers()), { "content-type": "application/json" }), e4.headers), queryParameters: r(r({}, n3.queryParameters()), e4.queryParameters) }));
              return h({ appId: e4.appId, transporter: a3 }, e4.methods);
            }(r(r(r({}, f2), e3), {}, { methods: { getPersonalizationStrategy: W, setPersonalizationStrategy: H } }));
          };
        };
        if (n2 && n2.transformation) {
          if (!n2.transformation.region) throw qt("`region` must be provided when leveraging the transformation pipeline");
          o2 = function(e3) {
            if (!e3 || !e3.transformation || !e3.transformation.region) throw qt("`region` must be provided when leveraging the transformation pipeline");
            if ("eu" !== e3.transformation.region && "us" !== e3.transformation.region) throw qt("`region` is required and must be one of the following: eu, us");
            var t3 = e3.appId, n3 = c(m.WithinHeaders, t3, e3.apiKey), a3 = k(r(r({ hosts: [{ url: "data.".concat(e3.transformation.region, ".algolia.com"), accept: y.ReadWrite, protocol: "https" }] }, e3), {}, { headers: r(r(r({}, n3.headers()), { "content-type": "text/plain" }), e3.headers), queryParameters: r(r({}, n3.queryParameters()), e3.queryParameters) }));
            return { transporter: a3, appId: t3, addAlgoliaAgent: function(e4, t4) {
              a3.userAgent.add({ segment: e4, version: t4 }), a3.userAgent.add({ segment: "Ingestion", version: t4 }), a3.userAgent.add({ segment: "Ingestion via Algoliasearch" });
            }, clearCache: function() {
              return Promise.all([a3.requestsCache.clear(), a3.responsesCache.clear()]).then(function() {
              });
            }, push: function(e4, t4) {
              var n4 = e4.indexName, o3 = e4.pushTaskPayload, i2 = e4.watch;
              if (!n4) throw qt("Parameter `indexName` is required when calling `push`.");
              if (!o3) throw qt("Parameter `pushTaskPayload` is required when calling `push`.");
              if (!o3.action) throw qt("Parameter `pushTaskPayload.action` is required when calling `push`.");
              if (!o3.records) throw qt("Parameter `pushTaskPayload.records` is required when calling `push`.");
              var u2 = t4 || { queryParameters: {} };
              return a3.write({ method: j, path: p("1/push/%s", n4), data: o3 }, r(r({}, u2), {}, { queryParameters: r(r({}, u2.queryParameters), {}, { watch: void 0 !== i2 }) }));
            } };
          }(r(r({}, n2), f2));
        }
        return function(e3) {
          var t3 = e3.appId, n3 = c(void 0 !== e3.authMode ? e3.authMode : m.WithinHeaders, t3, e3.apiKey), a3 = k(r(r({ hosts: [{ url: "".concat(t3, "-dsn.algolia.net"), accept: y.Read }, { url: "".concat(t3, ".algolia.net"), accept: y.Write }].concat(l([{ url: "".concat(t3, "-1.algolianet.com") }, { url: "".concat(t3, "-2.algolianet.com") }, { url: "".concat(t3, "-3.algolianet.com") }])) }, e3), {}, { headers: r(r(r({}, n3.headers()), { "content-type": "application/x-www-form-urlencoded" }), e3.headers), queryParameters: r(r({}, n3.queryParameters()), e3.queryParameters) }));
          return h({ transporter: a3, appId: t3, addAlgoliaAgent: function(e4, t4) {
            a3.userAgent.add({ segment: e4, version: t4 });
          }, clearCache: function() {
            return Promise.all([a3.requestsCache.clear(), a3.responsesCache.clear()]).then(function() {
            });
          } }, e3.methods);
        }(r(r({}, d2), {}, { methods: { search: me, searchForFacetValues: ge, multipleBatch: he, multipleGetObjects: pe, multipleQueries: me, copyIndex: Q, copySettings: _, copySynonyms: X, copyRules: V, moveIndex: le, listIndices: fe, getLogs: ne, listClusters: ce, multipleSearchForFacetValues: ge, getApiKey: ee, addApiKey: M, listApiKeys: se, updateApiKey: xe, deleteApiKey: Z, restoreApiKey: be, assignUserID: B, assignUserIDs: G, getUserID: oe, searchUserIDs: Oe, listUserIDs: de, getTopUserIDs: ae, removeUserID: ye, hasPendingMappings: ie, clearDictionaryEntries: L, deleteDictionaryEntries: $, getDictionarySettings: re, getAppTask: te, replaceDictionaryEntries: ve, saveDictionaryEntries: we, searchDictionaryEntries: Pe, setDictionarySettings: Ie, waitAppTask: je, customRequest: Y, initIndex: function(e3) {
          return function(t3) {
            return r(r({}, ue(e3)(t3, { methods: { batch: qe, delete: Ce, findAnswers: He, getObject: Me, getObjects: Ge, saveObject: et, saveObjects: tt, search: it, searchForFacetValues: ut, waitTask: dt, setSettings: ft, getSettings: Qe, partialUpdateObject: _e, partialUpdateObjects: Xe, deleteObject: Ue, deleteObjects: ze, deleteBy: Re, clearObjects: Ne, browseObjects: De, getObjectPosition: Be, findObject: Ke, exists: We, saveSynonym: at, saveSynonyms: ot, getSynonym: Ve, searchSynonyms: ct, browseSynonyms: ke, deleteSynonym: Fe, clearSynonyms: Ae, replaceAllObjects: Ye, replaceAllSynonyms: $e, searchRules: st, getRule: Le, deleteRule: Je, saveRule: rt, saveRules: nt, replaceAllRules: Ze, browseRules: Te, clearRules: Ee } })), {}, { saveObjectsWithTransformation: xt(t3, o2), partialUpdateObjectsWithTransformation: jt(t3, o2) });
          };
        }, initAnalytics: function() {
          return function(e3) {
            return function(e4) {
              var t3 = e4.region || "us", n3 = c(m.WithinHeaders, e4.appId, e4.apiKey), a3 = k(r(r({ hosts: [{ url: "analytics.".concat(t3, ".algolia.com") }] }, e4), {}, { headers: r(r(r({}, n3.headers()), { "content-type": "application/json" }), e4.headers), queryParameters: r(r({}, n3.queryParameters()), e4.queryParameters) }));
              return h({ appId: e4.appId, transporter: a3 }, e4.methods);
            }(r(r(r({}, f2), e3), {}, { methods: { addABTest: C, getABTest: z, getABTests: J, stopABTest: F, deleteABTest: U } }));
          };
        }, initPersonalization: g2, initRecommendation: function() {
          return function(e3) {
            return d2.logger.info("The `initRecommendation` method is deprecated. Use `initPersonalization` instead."), g2()(e3);
          };
        }, getRecommendations: yt, getFrequentlyBoughtTogether: vt, getLookingSimilar: Ot, getRecommendedForYou: It, getRelatedProducts: bt, getTrendingFacets: wt, getTrendingItems: Pt } }));
      }
      return Dt.version = "4.25.2", Dt;
    });
  }
});
export default require_algoliasearch_umd();
/*! Bundled license information:

algoliasearch/dist/algoliasearch.umd.js:
  (*! algoliasearch.umd.js | 4.25.2 | © Algolia, inc. | https://github.com/algolia/algoliasearch-client-javascript *)
*/
//# sourceMappingURL=algoliasearch.js.map
