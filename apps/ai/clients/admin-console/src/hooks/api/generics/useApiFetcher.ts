import { API_URL } from '@/config'
import { useAuth } from '@/contexts/auth-context'
import {
  ESubscriptionErrorCode,
  isSubscriptionErrorCode,
  useSubscription,
} from '@/contexts/subscription-context'
import { useRouter } from 'next/navigation'
import { useCallback, useState } from 'react'

const useApiFetcher = () => {
  const { token, fetchToken } = useAuth()
  const { setSubscriptionStatus } = useSubscription()
  const router = useRouter()

  const [controller, setController] = useState(new AbortController())

  const apiFetcher = useCallback(
    async <T>(url: string, options?: RequestInit, retry = true): Promise<T> => {
      if (!token) return Promise.resolve(null as unknown as T)
      const headers = {
        Authorization: `Bearer ${token}`,
        ...(!(options?.body instanceof FormData)
          ? { 'Content-Type': 'application/json' }
          : {}),
        ...(options?.headers || {}),
      }

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers,
      })

      if (!response.ok) {
        if (response.status === 401) {
          if (retry === false) {
            console.error(`Authentication failed. Redirecting to login page...`)
            router.push('api/auth/logout')
          }
          try {
            await fetchToken()
            return apiFetcher<T>(url, { ...options }, false)
          } catch (e) {
            console.error(
              `Authentication failed: ${e}. Redirecting to login page...`,
            )
            router.push('api/auth/logout')
          }
        } else {
          const serverError: { detail: string } = await response.json()
          const error = new Error(serverError.detail, {
            cause: response.status,
          })
          // for now the error codes are in the `detail` field. This will become a JSON in the future
          if (isSubscriptionErrorCode(serverError.detail)) {
            setSubscriptionStatus(serverError.detail as ESubscriptionErrorCode)
          }

          throw error
        }
      }
      return response.json()
    },
    [controller.signal, fetchToken, router, setSubscriptionStatus, token],
  )

  const cancelApiFetch = useCallback(() => {
    controller.abort()
    setController(new AbortController())
  }, [controller])

  const apiDownloadFile = async (endpointUrl: string): Promise<Blob | null> => {
    try {
      const response = await fetch(`${API_URL}/${endpointUrl}`, {
        method: 'GET',
        headers: { Authorization: `Bearer ${token}` },
      })

      if (!response.ok) {
        console.error('Download error:', response.statusText)
        throw new Error(response.statusText)
      }

      return await response.blob()
    } catch (error) {
      console.error('Download error:', error)
      throw error
    }
  }

  return { apiFetcher, cancelApiFetch, apiDownloadFile }
}

export default useApiFetcher
