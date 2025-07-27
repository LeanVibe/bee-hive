declare namespace NodeJS {
  interface Timeout {
    ref(): this
    unref(): this
  }
}